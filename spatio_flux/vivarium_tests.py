from vivarium import Vivarium
from spatio_flux import PROCESS_DICT, TYPES_DICT

def run_vivarium_dfba():
    v = Vivarium(processes=PROCESS_DICT, types=TYPES_DICT)

    # add a dynamic FBA process called 'dFBA'
    v.add_process(name="dFBA",
                  process_id="DynamicFBA",
                  config={
                      "model_file": "textbook",
                      "kinetic_params": {
                          "glucose": (0.5, 1),
                          "acetate": (0.5, 2)},
                      "substrate_update_reactions": {
                          "glucose": "EX_glc__D_e",
                          "acetate": "EX_ac_e"},
                      "biomass_identifier": "biomass",
                      "bounds": {
                          "EX_o2_e": {"lower": -2, "upper": None},
                          "ATPM": {"lower": 1, "upper": 1}}},
                  )


    # v.add_object(name="fields", type="array")
    v.connect_process(
        process_name="dFBA",
        inputs={
                "substrates": ["fields", 0],  # {mol_id: ['fields', mol_id] for mol_id in mol_ids}
            },
        outputs={
                "substrates": ["fields", 0],  # {mol_id: ['fields', mol_id] for mol_id in mol_ids}
            }
    )

    breakpoint()

if __name__ == '__main__':
    run_vivarium_dfba()
