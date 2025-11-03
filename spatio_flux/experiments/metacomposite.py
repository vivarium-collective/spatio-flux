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
        schema, state = core.generate({}, doc)
        # full_state = core.fill(schema, state)

        inputs = schema[process_name]['_inputs']
        outputs = schema[process_name]['_outputs']
        state[process_name]['_type'] = 'process'
        state[process_name]['_inputs'] = {}
        state[process_name]['_outputs'] = {}
        state[process_name]['inputs'] = {}
        state[process_name]['outputs'] = {}

        # add inputs and outputs port schemas to state for visualization
        for input_name, input_schema in inputs.items():
            state[process_name]['_inputs'].update({input_name: input_schema})
        for output_name, output_schema in outputs.items():
            state[process_name]['_outputs'].update({output_name: output_schema})

        fname = f"{process_name}_disconnected"
        plot_bigraph(
            state=state,
            core=core,
            out_dir=str(outdir),
            filename=fname,
            dpi="300",
            collapse_redundant_processes=True,
        )

        # add inputs and outputs port connections to state for visualization
        for input_name, input_schema in inputs.items():
            state[process_name]['inputs'].update({input_name: [input_name]})
        for output_name, output_schema in outputs.items():
            state[process_name]['outputs'].update({output_name: [output_name]})

        fname = f"{process_name}_connected"
        plot_bigraph(
            state=state,
            core=core,
            out_dir=str(outdir),
            filename=fname,
            dpi="300",
            show_types=True, # this helps see the type information in the stores
            collapse_redundant_processes=True,
        )



if __name__ == "__main__":
    main()