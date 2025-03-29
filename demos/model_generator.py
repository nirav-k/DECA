import subprocess

def run_script(script_name, args):
    cmd = ["python", script_name] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"{script_name} completed successfully!\n")
    else:
        print(f"Error running {script_name}:")
        print(result.stderr)
        exit(1)  # Exit if a script fails

if __name__ == "__main__":
    # Define arguments for each script
    common_args = [
        "--image_path", "TestSamples/examples/Mit/Mit cropped.jpg",
        "--exp_path", "TestSamples/exp/",
        "--savefolder", "output/",
        "--rasterizer_type", "pytorch3d",
        "--saveObj", "True"
    ]

    # Run the first script (Expression Transfer)
    run_script("demos/neutral_transfer.py", common_args + ["--useTex", "True"])

    # Run the second script (Processing & Saving)
    run_script("demos//vowel_transfer.py", common_args + ["--useTex", "False"])