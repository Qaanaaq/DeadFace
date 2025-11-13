import csv

# Create CSV and write the header
def initialize_csv(output_path, blendshape_names):
    header = ["Timecode", "BlendShapeCount"] + blendshape_names
    csv_file = open(output_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(header)
    return csv_file, writer

# Write a single row of blendshape data
def write_blendshape_row(writer, timecode, num_blendshapes, blendshapes_sorted, tongue, head_rotation, eyes):
    row = [timecode] + [num_blendshapes] + blendshapes_sorted + tongue + head_rotation + eyes
    writer.writerow(row)

# Close the CSV when done
def close_csv(csv_file):
    csv_file.close()
