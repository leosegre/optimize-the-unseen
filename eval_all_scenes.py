import os
import subprocess
import argparse


def run_command(command):
    """Helper function to run a command and handle errors."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        exit(result.returncode)


def main(data_dir, masks_dir, run_gs, method, scene_names, measure_time, dataparser):
    # Iterate over each scene subdirectory
    for scene_name in os.listdir(data_dir):
        if scene_names is not None:
            if scene_name not in scene_names:
                continue
        scene_path = os.path.join(data_dir, scene_name)

        data_name = scene_name
        scene_name = scene_name
        measure_time = "--measure-time" if measure_time else ""

        results_dir = method

        if method == "splatfacto":
            timestamp = "base"
            command_extras = ""
        elif method == "clean-splatfacto":
            timestamp = "clean"
            # command_extras = "--pipeline.model.area-reg True --pipeline.model.opacity-regularization True"
            # command_extras = "--pipeline.model.scale-reg-simple True --pipeline.model.opacity-regularization True"
            command_extras = "--pipeline.model.opacity-regularization True --pipeline.model.sh-reg True"
            # command_extras = "--pipeline.model.mean-reg True"
            # command_extras = "--pipeline.model.opacity-regularization True --pipeline.model.mean-reg True --pipeline.model.area-reg True"

            command_extras = f"{command_extras} --pipeline.model.reset_alpha_every 10000"

        if dataparser == "nb-dataparser":
            dataparser_cmd = f"nb-dataparser --eval-mode eval-frame-index --train-frame-indices 0 --eval-frame-indices 1 --downscale-factor 2"
        elif dataparser == "nerfstudio-data":
            dataparser_cmd = f"nerfstudio-data"
        elif dataparser == "colmap":
            dataparser_cmd = f"colmap --downscale-rounding-mode ceil --colmap-path sparse/0"

        if os.path.isdir(scene_path):
            # Step 1: Optionally run the NeRF
            if run_gs:
                gs_command = (
                    f"ns-train splatfacto-mcmc --vis viewer --data {data_dir}/{data_name}/ --pipeline.model.camera-optimizer.mode off "
                    f"--experiment-name {scene_name} --output-dir outputs --timestamp {timestamp} --viewer.quit-on-train-completion True "
                    f"--relative-model-dir=nerfstudio_models/ --max-num-iterations=30000 --pipeline.model.background-color random "
                    f"--pipeline.model.rasterize-mode antialiased "
                    f"--pipeline.model.stop-split-at 15000 "
                    # f"--pipeline.model.reset_alpha_every 10000 "
                    # f"--pipeline.model.random-init True "
                    f"--pipeline.model.cull_alpha_thresh=0.005 "
                    f"--pipeline.model.use_scale_regularization True "
                    # f"--pipeline.model.color-corrected-metrics True "
                    f"--logging.local-writer.enable False "
                    # f"--pipeline.datamanager.train-cameras-sampling-strategy fps "
                    f"--viewer.websocket-port 7008 "
                    f"{command_extras} "
                    f"{dataparser_cmd} "
                    # f"nb-dataparser --eval-mode eval-frame-index --train-frame-indices 0 --eval-frame-indices 1 --downscale-factor 2"
                )
                run_command(gs_command)

            # Step 2: Run the evaluation
            if masks_dir is None:
                masks_cmd = ""
            else:
                masks_cmd = f"--nb-mask --visibility-path {masks_dir}/{data_name}"

            eval_command = (
                f"ns-eval-cleanup --load-config outputs/{scene_name}/splatfacto/{timestamp}/config.yml "
                f"--output-path results/{results_dir}/{scene_name}.json "
                f"{masks_cmd} "
                f"{measure_time} "
            )
            run_command(eval_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeRF and evaluation for each scene.")
    parser.add_argument("data_dir", help="Path to the data directory.")
    parser.add_argument("masks_dir", help="Path to the masks directory.")
    parser.add_argument("--run-gs", action="store_true",
                        help="Flag to indicate whether to run the GS training step.")
    parser.add_argument("--method", help="Method to run.", choices=["splatfacto", "clean-splatfacto"], default="clean-splatfacto")
    parser.add_argument("--scene-names", nargs="+", default=None, help="List of scene names to evaluate.")
    parser.add_argument("--measure-time", action="store_true", help="Flag to measure the time taken for each step.")
    parser.add_argument("--dataparser", choices=["nb-dataparser", "nerfstudio-data", "colmap"], default="nb-dataparser", help="Dataparser to use for evaluation.")
    args = parser.parse_args()

    main(args.data_dir, args.masks_dir, args.run_gs, args.method, args.scene_names, args.measure_time, args.dataparser)
