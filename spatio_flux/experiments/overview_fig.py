from process_bigraph import register_types as register_process_types
from vivarium.vivarium import VivariumTypes
from bigraph_viz import plot_bigraph
from spatio_flux import register_types


def get_dfba_single_doc(
        core=None,
        config=None,
):
    return {
        "_type": "process",
        "address": "local:DynamicFBA",
        "config": {'model_file': 'textbook'},
        # "inputs": {
        #     "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids},
        #     "biomass": build_path(path, biomass_id, i, j)
        # },
        # "outputs": {
        #     "substrates": {mol_id: build_path(path, mol_id, i, j) for mol_id in mol_ids},
        #     "biomass": build_path(path, biomass_id, i, j)
        # }
    }

PROCESS_DOCS = {
    'dfba_single': get_dfba_single_doc,
}

def main():
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)


    for name, get_doc in PROCESS_DOCS.items():
        document = get_doc(core=core)
        # composite = Composite(document=document, core=core)
        # plot_state = gather_emitter_results(
        #     composite=composite,
        #     emitter=core.get_emitter('default'),
        #     time=0,
        # )
        # plot_schema = composite.get_schema()


        plot_bigraph(
            state=document,
            # schema=plot_schema,
            core=core,
            out_dir='out',
            filename=f'{name}_viz',
            dpi='300',
            collapse_redundant_processes=True
        )



if __name__ == '__main__':
    main()
