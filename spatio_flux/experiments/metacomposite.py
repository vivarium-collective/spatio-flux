from vivarium.vivarium import VivariumTypes
from process_bigraph import register_types as register_process_types
from spatio_flux.processes import PROCESS_DICT, PROCESS_DOCS
from spatio_flux import register_types, SPATIO_FLUX_TYPES


def main():

    # establish core
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    print("Registered Processes:")
    for process_name, get_doc in PROCESS_DOCS.items():
        print(f"- {process_name}")
        doc = get_doc(core=core)
        print(doc)

        schema, state = core.generate({}, doc)
        breakpoint()



if __name__ == "__main__":
    main()