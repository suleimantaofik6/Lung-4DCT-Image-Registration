import os
import tempfile
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

class RawToITKConverter:
    def __init__(self):
        self.pixel_dict = {
            sitk.sitkUInt8: "MET_UCHAR",
            sitk.sitkInt8: "MET_CHAR",
            sitk.sitkUInt16: "MET_USHORT",
            sitk.sitkInt16: "MET_SHORT",
            sitk.sitkUInt32: "MET_UINT",
            sitk.sitkInt32: "MET_INT",
            sitk.sitkUInt64: "MET_ULONG_LONG",
            sitk.sitkInt64: "MET_LONG_LONG",
            sitk.sitkFloat32: "MET_FLOAT",
            sitk.sitkFloat64: "MET_DOUBLE",
        }
        self.direction_cosine = [
            "1 0 0 1",
            "1 0 0 0 1 0 0 0 1",
            "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
        ]

    def convert_raw_to_ITK_readable(
        self,
        binary_file_name,
        image_size,
        output_path,
        sitk_pixel_type=sitk.sitkInt16,
        image_spacing=None,
        image_origin=None,
        big_endian=False,
    ):
        dim = len(image_size)
        header = [
            "ObjectType = Image\n".encode(),
            (f"NDims = {dim}\n").encode(),
            (
                "DimSize = "
                + " ".join([str(v) for v in image_size])
                + "\n"
            ).encode(),
            (
                "ElementSpacing = "
                + (
                    " ".join([str(v) for v in image_spacing])
                    if image_spacing
                    else " ".join(["1"] * dim)
                )
                + "\n"
            ).encode(),
            (
                "Offset = "
                + (
                    " ".join([str(v) for v in image_origin])
                    if image_origin
                    else " ".join(["0"] * dim) + "\n"
                )
            ).encode(),
            ("TransformMatrix = " + self.direction_cosine[dim - 2] + "\n").encode(),
            ("ElementType = " + self.pixel_dict[sitk_pixel_type] + "\n").encode(),
            "BinaryData = True\n".encode(),
            ("BinaryDataByteOrderMSB = " + str(big_endian) + "\n").encode(),
            (
                "ElementDataFile = " + os.path.abspath(binary_file_name) + "\n"
            ).encode(),
        ]

        fp = tempfile.NamedTemporaryFile(suffix=".mhd", delete=False)
        fp.writelines(header)
        fp.close()
        img = sitk.ReadImage(fp.name)
        os.remove(fp.name)
        
        # Save the NIfTI image
        # sitk.WriteImage(img, output_path)

        return img

    def convert_raw_files_to_nifti(
        self, folder, output_folder, image_size, sitk_pixel_type=sitk.sitkInt16
    ):
        for file_name in os.listdir(folder):
            if file_name.endswith(".img"):
                raw_path = os.path.join(folder, file_name)
                output_path = os.path.join(
                    output_folder, file_name.replace(".img", ".nii")
                )

                # Convert the raw image to NIfTI
                self.convert_raw_to_ITK_readable(
                    binary_file_name=raw_path,
                    image_size=image_size,
                    output_path=output_path,
                )

    def visualize_all_position(self, folder_path):
        nifty_files = [
            file for file in os.listdir(folder_path) if file.endswith(".nii")
        ]

        for nifty_file in nifty_files:
            file_path = os.path.join(folder_path, nifty_file)
            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image_array[60, :, :], cmap="gray")
            axes[0].set_title("Axial Slice at Middle Index")

            axes[1].imshow(
                image_array[:, image_array.shape[1] // 2, :], cmap="gray"
            )
            axes[1].set_title("Sagittal Slice at Middle Index")

            axes[2].imshow(
                image_array[:, :, image_array.shape[2] // 2], cmap="gray"
            )
            axes[2].set_title("Coronal Slice at Middle Index")

            plt.suptitle(f"Visualization for {nifty_file}")
            plt.show()

    def visualize_with_landmarks(self, folder_path):
        nifty_files = [
            file for file in os.listdir(folder_path) if file.endswith(".nii")
        ]

        for i in range(0, len(nifty_files), 2):
            exhale_file = nifty_files[i]
            inhale_file = nifty_files[i + 1]

            inhale_file_path = os.path.join(folder_path, inhale_file)
            exhale_file_path = os.path.join(folder_path, exhale_file)

            inhale_image = sitk.ReadImage(inhale_file_path)
            exhale_image = sitk.ReadImage(exhale_file_path)
            inhale_array = sitk.GetArrayFromImage(inhale_image)
            exhale_array = sitk.GetArrayFromImage(exhale_image)

            case_identifier = inhale_file.split("_")[0]
            inhale_exhale = inhale_file.split("_")[1][0].lower()
            inhale_landmark_file_name = f"{case_identifier}_300_iBH_xyz_r1.txt"
            exhale_landmark_file_name = f"{case_identifier}_300_eBH_xyz_r1.txt"
            inhale_landmark_file_path = os.path.join(
                folder_path, inhale_landmark_file_name
            )
            exhale_landmark_file_path = os.path.join(
                folder_path, exhale_landmark_file_name
            )

            inhale_landmarks = np.loadtxt(inhale_landmark_file_path)
            exhale_landmarks = np.loadtxt(exhale_landmark_file_path)

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            axes[0].imshow(inhale_array[60, :, :], cmap="gray")
            axes[0].scatter(
                inhale_landmarks[:, 0],
                inhale_landmarks[:, 1],
                c="red",
                marker="o",
                label="Inhale Landmarks",
            )
            axes[0].set_title("Inhale Axial Slice at Middle Index")

            axes[1].imshow(exhale_array[60, :, :], cmap="gray")
            axes[1].scatter(
                exhale_landmarks[:, 0],
                exhale_landmarks[:, 1],
                c="blue",
                marker="o",
                label="Exhale Landmarks",
            )
            axes[1].set_title("Exhale Axial Slice at Middle Index")

            plt.suptitle(f"Visualization for {case_identifier}")
            plt.legend()
            plt.show()


# =====================
# TRE IMPLEMENTATION
# =====================
def compute_tre(inhale_landmarks, exhale_landmarks, voxel_size):
    
    # Normalize landmarks using voxel size
    normalized_inhale_landmarks = inhale_landmarks * voxel_size
    normalized_exhale_landmarks = exhale_landmarks * voxel_size

    # Compute Euclidean distance between corresponding normalized landmarks
    landmark_distances = np.linalg.norm(normalized_inhale_landmarks - normalized_exhale_landmarks, axis=1)

    # Compute TRE (mean and standard deviation)
    mean_tre = np.mean(landmark_distances)
    std_tre = np.std(landmark_distances)

    return mean_tre, std_tre

def compare_landmarks(folder_path, voxel_sizes):
    # Get a list of all .txt files in the folder
    landmark_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    # Iterate over pairs of inhale and exhale files
    for inhale_file in landmark_files:
        # Check if the file corresponds to inhale based on the file name
        if "iBH" in inhale_file:
            case_identifier = inhale_file.split('_')[0]

            # Generate the corresponding exhale file name
            exhale_landmark_file_name = inhale_file.replace("_iBH_", "_eBH_")
            exhale_landmark_file_path = os.path.join(folder_path, exhale_landmark_file_name)

            # Check if the corresponding exhale file exists
            if os.path.exists(exhale_landmark_file_path):
                # Read the inhale and exhale landmark files
                inhale_landmarks = np.loadtxt(os.path.join(folder_path, inhale_file))
                exhale_landmarks = np.loadtxt(exhale_landmark_file_path)
                

                # Get voxel size for the case
                voxel_size = voxel_sizes.get(case_identifier)

                if voxel_size is not None:
                    # Compute TRE using voxel size
                    mean_tre, std_tre = compute_tre(inhale_landmarks, exhale_landmarks, voxel_size)

                    # Print the results
                    print(f'TRE for {case_identifier}:')
                    print(f'Mean TRE: {mean_tre:.2f} mm')
                    print(f'Std TRE: {std_tre:.2f} mm\n')
                else:
                    print(f'Voxel size not found for {case_identifier}.\n')
            else:
                print(f'Exhale landmark file not found for {case_identifier}.\n')


# =======================================================================
# CLEANING THE OUTPUT PARAMETER FILE AND EXTRACT ONLY THE OUTPUT POINTS
# ========================================================================
def clean_output_points(input_file_path, output_file_path):
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()

    output_points = []

    for line in lines:
        if 'OutputPoint' in line:
            point_data = [float(coord) for coord in line.split('; OutputPoint = [')[1].split(']')[0].split()]
            output_points.append(point_data)

    with open(output_file_path, 'w') as outfile:
        for point in output_points:
            outfile.write(f"{point[0]:.6f}\t{point[1]:.6f}\t{point[2]:.6f}\n")

    print("Output points extracted and saved to", output_file_path)
