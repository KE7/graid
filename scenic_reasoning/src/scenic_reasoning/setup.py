import subprocess
import os

def install_depth_pro():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, '..', '..', '..')

    os.chdir(root_dir)

    # Check if the repository already exists in the root
    if not os.path.exists('ml-depth-pro'):
        subprocess.run(['git', 'clone', 'https://github.com/apple/ml-depth-pro'])
    
    # Change to the ml-depth-pro directory
    os.chdir('ml-depth-pro')
    
    # Install the package in editable mode
    subprocess.run(['pip', 'install', '-e', '.'])

    # Change back to the original directory
    os.chdir('..')
    
    # Check if the checkpoints folder exists and contains depth_pro.pt
    if not (os.path.exists('checkpoints') and os.path.isfile('checkpoints/depth_pro.pt')):
        # Run the get_pretrained_models.sh script
        subprocess.run(['bash', 'ml-depth-pro/get_pretrained_models.sh'])
    