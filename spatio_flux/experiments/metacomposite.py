from bigraph_viz import plot_bigraph
from vivarium.vivarium import VivariumTypes
from process_bigraph import register_types as register_process_types
from spatio_flux.processes import PROCESS_DICT, PROCESS_DOCS
from spatio_flux import register_types, SPATIO_FLUX_TYPES


def main():

    outdir = "out/metacomposite"

    # establish core
    core = VivariumTypes()
    core = register_process_types(core)
    core = register_types(core)

    print("Registered Processes:")
    for process_name, get_doc in PROCESS_DOCS.items():
        print(f"- {process_name}")
        doc = get_doc(core=core)
        print(doc)

        # full_state = core.fill({}, doc)
        schema, doc = core.generate({}, doc)
        # full_state = core.fill(schema, state)

        inputs = schema[process_name]['_inputs']
        outputs = schema[process_name]['_outputs']
        doc[process_name]['_inputs'] = {}
        doc[process_name]['_outputs'] = {}
        doc[process_name]['inputs'] = {}
        doc[process_name]['outputs'] = {}

        # add inputs and outputs types
        for input_name, input_schema in inputs.items():
            doc[process_name]['_inputs'].update({input_name: input_schema})
            # state[process_name]['inputs'].update({input_name: [input_name]})
        for output_name, output_schema in outputs.items():
            doc[process_name]['_outputs'].update({output_name: output_schema})
            # state[process_name]['outputs'].update({output_name: [output_name]})

        fname = f"{process_name}_process"
        plot_bigraph(
            state=doc,
            core=core,
            out_dir=str(outdir),
            filename=fname,
            dpi="300",
            collapse_redundant_processes=True,
        )
        # png = outdir / f"{fname}.png"


        # breakpoint()
            # top_schema, top_state = core.infer_wires(input_schema, state, doc)




if __name__ == "__main__":
    main()