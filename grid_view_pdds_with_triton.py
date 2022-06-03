from pdds.grid_view_pdds import grid_view_pdds

def build_anomaly_analysis_graph(
    space_id,
    date_str,
    batch_size=128,
    slice_size=1024,
    cluster_eps=1000,
    cluster_min_samples=4,
    results_output_folder_path="/cvf/temp",
):
    graph = Graph()
    # Load local image
    graph.add_component("grid_view_pdds", grid_view_pdds)
    # graph.add_component('results_collector', data_results_collector)
    graph.initialize(False, "grid_view_pdds.debug_in")
    graph.initialize(results_output_folder_path, "grid_view_pdds.debug_folder_in")
    graph.initialize(space_id, "grid_view_pdds.space_id_in")
    graph.initialize(slice_size, "grid_view_pdds.slice_width_in")
    graph.initialize(slice_size, "grid_view_pdds.slice_height_in")
    graph.initialize(cluster_eps, "grid_view_pdds.cluster_eps_in")
    graph.initialize(cluster_min_samples, "grid_view_pdds.cluster_min_samples_in")
    graph.initialize(date_str, "grid_view_pdds.date_str_in")
    graph.initialize(batch_size, "grid_view_pdds.batch_size_in")
    # graph.initialize('results', 'results_collector.result_name_in')
    # graph.connect('grid_view_pdds.data_out', 'results_collector.data_in')
    return graph


def get_full_space_pdds_predictions(
    space_id,
    date_str,
    batch_size=128,
    results_output_folder_path="/workspaces/cvf/temp",
):
    graph = build_anomaly_analysis_graph(
        space_id, date_str, results_output_folder_path=results_output_folder_path
    )
    graph.export("grid_view_pdds.data_out", "full_space_pdds_preds")
    output = run_graph(graph, capture_results=True)
    return output["full_space_pdds_preds"]


if __name__ == "__main__":
    space_id = "75195"
    date_str = "07-08-2021"
    results_output_folder_path = (
        "/home/aboggaram/data/debug_images_full_space_pdds_07_08_2021_space_75195"
    )
    space_image_info_list = get_full_space_pdds_predictions(
        space_id,
        date_str,
        batch_size=128,
        results_output_folder_path=results_output_folder_path,
    )



if __name__ == "__main__":
    space_id = "75195"
    date_str = "07-08-2021"
    results_output_folder_path = (
        "/home/aboggaram/data/debug_images_full_space_pdds_07_08_2021_space_75195"
    )
    space_image_info_list = get_full_space_pdds_predictions(
        space_id,
        date_str,
        batch_size=128,
        results_output_folder_path=results_output_folder_path,
    )
