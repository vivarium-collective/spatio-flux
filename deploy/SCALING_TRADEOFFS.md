# Ray cluster scaling for exponentially-growing cell colonies

Tradeoff analysis: pre-allocate to peak vs dynamically scale, in the EC2
environment used for the COMETS comparison.

## Baseline (from `deploy/RUNNING_ON_EC2.md`)

- **Cluster**: head m5.2xlarge + 4× m5.4xlarge workers = **~$4.05/hr on-demand**, 64 worker vCPU
- **Cold cluster ready**: 5–10 min (best ~3 min after AMI bake), gated by EC2 launch + cloud-init, not Ray itself
- **Scale-up unit cost**: each new EC2 ≈ 3–5 min wall + AWS bills per-second (60s minimum)
- **Within-cluster scale-up**: `ActorPool.grow(n)` already exists — a few seconds, basically free if vCPU is idle

There are **two scaling timescales**:

1. **Actors on existing nodes** — seconds, free. Always do this.
2. **New EC2 nodes** — minutes, real money. The interesting tradeoff.

## Why exponential growth flips the usual intuition

Cell count `n(t) = n₀·2^(t/τ)`. Integrate over a run of duration T where
you end at N_peak:

- **Pre-allocate cost** ∝ N_peak · T
- **Perfect dynamic cost** ∝ ∫n(t)dt ≈ N_peak · τ / ln(2)
- **Ratio** = T·ln(2)/τ = **ln(N_peak/n₀)**

| Growth | Pre-alloc waste factor |
|---|---|
| 1 → 1k cells (10 doublings) | ~7× |
| 1 → 1M cells (20 doublings) | ~14× |

That sounds bad until you notice the other side: **half your total work
happens during the final doubling**. The last 3 doublings = 87.5% of
integrated compute. So even "perfect" dynamic scaling has the cluster
near peak size for most of the wall-clock anyway — you don't save as much
as the ratio suggests, because you can't usefully under-provision the
long tail at the end.

## Concrete cost comparison

4-hr sim, 1 → ~32k cells, the standard cluster:

| Strategy | Cost | Wall time | Notes |
|---|---|---|---|
| Pre-allocate 4 workers all 4hr | **$16.20** | 4hr + 5min boot | simplest, no surprises |
| Coarse autoscale (1→2→4 workers) | **~$11–12** | 4hr + 3×(3–5 min) under-provisioned phases | saves ~25–30%, costs ~15 min wall + complexity |
| Fine-grained autoscale | similar to coarse | worse wall time | every scale event eats 3–5 min; you spend more time waiting than computing during early phases |

The savings are real but modest — maybe **$3–5 on a $16 run**. You pay
it back in (a) wall-clock for under-provisioned bootstrap windows and
(b) engineering complexity (`EC2SSMRayCluster` is currently a
static-size context manager; would need an `add_workers()` path).

## What changes the math

- **Spot instances**: m5.4xlarge spot is typically 60–70% off → over-provisioning gets cheaper, autoscaling gets less attractive (and spot loss = shard restart pain).
- **Long sims (days, not hours)**: bootstrap amortizes to ~nothing; pre-allocation tax dominates → autoscaling wins.
- **Memory-bound, not CPU-bound**: if cells need to live in shard state, you might be forced to scale nodes for RAM regardless of compute. m5.4xlarge has 64 GB.
- **Truly unbounded growth**: you can't pre-allocate. But "unbounded" usually means "you should checkpoint and run bounded chunks" rather than "you need an autoscaler."

## Recommendation

1. **Bounded peak (you can guess N_peak within ~4×)**: pre-allocate to peak. The math says wasted compute is small relative to engineering cost of dynamic scaling.
2. **Within-node growth**: lean hard on `ActorPool.grow()` — already cheap, worth using aggressively.
3. **Genuinely unbounded**: don't autoscale continuously — pick coarse cluster sizes (1×, 2×, 4× workers) and double when you cross a vCPU threshold. That bounds scale-up events to ~log₂(growth) and keeps the autoscaler logic simple. Would need an `add_workers()` path on `EC2SSMRayCluster` upstream.
4. **Sanity check first**: profile a short run with the static cluster and look at `ray status` utilization. If average utilization is >40–50%, autoscaling is just churn. For exponential colony growth, expect that to be the common case.

**The non-obvious takeaway**: for exponential growth, "pay for peak the
whole time" is usually within 2–3× of optimal *in dollars*, while being
~10× simpler operationally. Autoscaling is much more compelling for
*bursty* or *flat* workloads than for monotonic exponentials.
