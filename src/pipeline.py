import argparse

from data_loader import DataLoader
from imputer import DataImputer
from visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing the data files.",
    )
    args = parser.parse_args()

    # Initialize the DataLoader with the input directory
    data_loader = DataLoader(args.input_dir)
    # Load the data and feature map
    data_dict, feature_map = data_loader.get_dataframe()

    # Impute relevant DFs - stop_it, tap_it
    stop_it_imputer = DataImputer(data_dict["stop_it"], feature_map)
    data_dict["stop_it"] = stop_it_imputer.impute_and_build()

    tap_it_imputer = DataImputer(data_dict["tap_it"], feature_map)
    data_dict["tap_it"] = tap_it_imputer.impute_and_build()

    # Initialize the Visualizer with the original and imputed data
    visualizer = Visualizer(data_dict)
    # Visualize some samples
    visualizer.plot_task_measure("stop_it", "SSRT")
    visualizer.plot_fluctuation_groups(
        "stop_it", "SSRT", "clinical", "AQ-BUSS_Physical-Aggression"
    )
    visualizer.plot_task_by_fluctuation_color("stop_it", "SSRT")
    visualizer.plot_mean_ci("tap_it", "A-cue bias")
    visualizer.plot_change_vs_arousal("tap_it", "A-cue bias", "stop_it", "SSRT")
    visualizer.plot_cross_domain_correlation(
        {
            "stop_it": ["SSRT", "Post Go Error Efficiency"],
            "tap_it": ["A-cue bias", "PBI_composite"],
        }
    )


if __name__ == "__main__":
    main()
