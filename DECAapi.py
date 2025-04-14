from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import shutil, time
import os, subprocess
import zipfile
from pathlib import Path

app = FastAPI()

UPLOAD_FOLDER = "upload"
OUTPUT_FOLDER = "output"
ZIP_FOLDER = "zips"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ZIP_FOLDER, exist_ok=True)

@app.post("/process/")
async def process_and_return(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    image_path = f"{UPLOAD_FOLDER}/{file.filename}"
    
    # Save the uploaded image
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image to generate .obj and .png
    process_image(image_path)  # Your function to generate files

    # Create a ZIP file with results
    zip_filename = f"{ZIP_FOLDER}/{Path(file.filename).stem}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for filename in os.listdir(OUTPUT_FOLDER):
            zipf.write(os.path.join(OUTPUT_FOLDER, filename), filename)

    # Schedule file cleanup in the background
    background_tasks.add_task(cleanup_files, image_path, OUTPUT_FOLDER, zip_filename)
    
    return FileResponse(zip_filename, media_type='application/zip', filename=f"{Path(file.filename).stem}.zip")


def process_image(image_path):
    common_args = [
        "--image_path", image_path,
        "--savefolder", "output/",
        "--rasterizer_type", "pytorch3d",
        "--saveObj", "True",
    ]

    # Run the first script (Expression Transfer)
    run_script("demos/custom_transfer.py", common_args + ["--exp_path", "TestSamples/exp/"] + ["--useTex", "False"])
    run_script("demos/custom_transfer_neutral.py", common_args + ["--useTex", "True"])

    os.remove("output/Face_Neutral_normals.png")
    os.remove("output/Face_Neutral.mtl")

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

def cleanup_files(image_path, output_folder, zip_path):
    """Deletes the uploaded image, generated files, and ZIP after response is sent."""
    time.sleep(5)  # Give some time to ensure response is fully sent

    # Delete uploaded image
    if os.path.exists(image_path):
        os.remove(image_path)

    # Delete output files
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    # Delete ZIP file
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"Deleted: {image_path}, {output_folder} contents, {zip_path}")

