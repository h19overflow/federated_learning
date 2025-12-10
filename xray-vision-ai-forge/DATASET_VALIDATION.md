# Dataset Validation Documentation

## Overview

The DatasetUpload component now includes comprehensive, interactive validation for chest X-ray datasets. The validation process ensures that your dataset structure, CSV format, and data integrity meet the requirements for successful training.

## Features

### 1. **Interactive Validation Process**

- Real-time progress updates as validation steps execute
- Visual indicators (pending, processing, success, warning, error) for each step
- Detailed messages and expandable details for transparency

### 2. **Comprehensive Checks**

#### Step 1: ZIP File Reading

- Validates the ZIP file can be opened
- Counts total files and directories

#### Step 2: Folder Structure Analysis

- Scans the entire ZIP structure
- Lists all directories found

#### Step 3: Images Directory Detection

- Searches for "Images" folder (case-insensitive)
- Works with nested structures
- Supports various image formats: .jpg, .jpeg, .png, .bmp, .gif
- Reports the exact path and number of images found

#### Step 4: CSV File Detection

- Searches for CSV files anywhere in the ZIP (can be nested)
- Handles multiple CSV files (uses the first one found with a warning)
- Reports the CSV file path

#### Step 5: CSV Format Validation

- Validates CSV structure using PapaParse
- **Required columns** (case-insensitive):
  - `PatientId` - Unique identifier for each patient
  - `Target` - Class label (e.g., NORMAL, PNEUMONIA)
- Reports any parsing warnings
- Shows total records and all column names

#### Step 6: Patient ID Matching

- Matches PatientId from CSV with image filenames
- Uses fuzzy matching (checks if PatientId is contained in filename or vice versa)
- Reports:
  - Total images found
  - Total CSV records
  - Successfully matched records
  - Unmatched Patient IDs
  - Match percentage

### 3. **Validation Results Summary**

After validation completes, a detailed summary shows:

#### Success Metrics

- **Images Found**: Total images in the Images directory
- **CSV Records**: Total rows in the CSV file
- **Matched**: Successfully matched Patient IDs
- **Classes**: Number of unique classes detected

#### Warnings

- Patient IDs without matching images (with percentage)
- Images without CSV entries
- Multiple CSV files found

#### Errors

- No Images directory found
- No CSV file found
- Missing required columns in CSV
- No matches between Patient IDs and images

#### Additional Information

- Images folder path
- CSV file path
- Detected classes
- List of unmatched Patient IDs (expandable)

## Dataset Structure Requirements

### Minimum Valid Structure

```
dataset.zip
├── Images/                    # Required (case-insensitive, can be nested)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── metadata.csv              # Required (can be nested)
```

### Example Valid Structures

#### Structure 1: Simple

```
dataset.zip
├── Images/
│   ├── NORMAL/
│   │   ├── patient001.jpg
│   │   └── patient002.jpg
│   └── PNEUMONIA/
│       ├── patient003.jpg
│       └── patient004.jpg
└── metadata.csv
```

#### Structure 2: Nested

```
dataset.zip
└── chest_xray_dataset/
    ├── data/
    │   └── Images/           # System finds this even when nested
    │       ├── patient001.jpg
    │       └── patient002.jpg
    └── labels/
        └── annotations.csv    # System finds this even when nested
```

#### Structure 3: With Additional Files

```
dataset.zip
├── README.md                  # Additional files are ignored
├── Images/
│   └── ...
├── metadata.csv
└── LICENSE.txt
```

### CSV Format

#### Required Columns (case-insensitive)

- `PatientId` or `patientid` or `PATIENTID`
- `Target` or `target` or `TARGET`

#### Example CSV Content

```csv
PatientId,Target,Age,Gender
patient001,NORMAL,45,M
patient002,NORMAL,32,F
patient003,PNEUMONIA,67,M
patient004,PNEUMONIA,54,F
```

**Note**: Additional columns (Age, Gender, etc.) are allowed and will be ignored during validation.

### Image Filename Matching

The system uses flexible matching between Patient IDs and image filenames:

✅ **Valid Matches**:

- PatientId: `patient001` → Image: `patient001.jpg`
- PatientId: `001` → Image: `patient_001.jpeg`
- PatientId: `ABC123` → Image: `ABC123_chest_xray.png`

❌ **Invalid Matches**:

- PatientId: `patient001` → Image: `unrelated.jpg`
- PatientId: `ABC` → Image: `XYZ123.jpg`

## User Interface

### Validation Progress Display

The validation steps are displayed in real-time with:

- **Pending**: Gray circle outline
- **Processing**: Spinning loader (blue)
- **Success**: Green checkmark
- **Warning**: Yellow warning triangle
- **Error**: Red X

### Interactive Features

1. **Show/Hide Details Button**: Toggle detailed information for each validation step
2. **Expandable Unmatched IDs**: Click to view the list of Patient IDs without matching images
3. **Retry Validation**: If validation fails, retry without re-uploading
4. **Progress Messages**: Each step shows contextual information

### Color-Coded Results

- **Success**: Green background with checkmark
- **Failure**: Red background with X icon
- **Warnings**: Yellow text and icons

## Error Handling

### Common Issues and Solutions

#### Issue 1: "Images directory not found"

**Cause**: No folder named "Images" exists in the ZIP file
**Solution**: Ensure your ZIP contains a folder named "Images" (case doesn't matter)

#### Issue 2: "No CSV file found"

**Cause**: No .csv file exists in the ZIP
**Solution**: Include a CSV file with your metadata

#### Issue 3: "Missing required columns"

**Cause**: CSV doesn't have PatientId or Target columns
**Solution**: Add these columns to your CSV:

```csv
PatientId,Target
patient001,NORMAL
patient002,PNEUMONIA
```

#### Issue 4: "No matches found between CSV and images"

**Cause**: Patient IDs don't match any image filenames
**Solution**: Ensure image filenames contain the Patient IDs from your CSV

#### Issue 5: High percentage of unmatched IDs

**Cause**: Some Patient IDs don't correspond to images
**Solution**:

- Check for typos in Patient IDs or filenames
- Ensure all images referenced in CSV are included in the ZIP
- **Note**: The system will still proceed if >90% of IDs match

## Technical Implementation

### Libraries Used

- **JSZip**: For reading and parsing ZIP files
- **PapaParse**: For robust CSV parsing with error handling

### Validation Flow

```
1. User uploads ZIP file
   ↓
2. Read ZIP contents
   ↓
3. Scan folder structure
   ↓
4. Find Images directory (case-insensitive, nested)
   ↓
5. Find CSV file (nested)
   ↓
6. Parse and validate CSV format
   ↓
7. Match Patient IDs with image filenames
   ↓
8. Generate validation report
   ↓
9. Display results to user
```

### Performance Considerations

- Large ZIP files (>1GB) may take longer to process
- Validation is performed client-side (no server upload required at this stage)
- Progress updates provide user feedback during processing

## Best Practices

1. **Organize your data**: Use clear folder structures and naming conventions
2. **Name consistency**: Ensure Patient IDs in CSV match image filenames
3. **File size**: Keep individual ZIP files under 5GB for optimal performance
4. **Image formats**: Use standard formats (JPEG, PNG) for best compatibility
5. **CSV cleanliness**: Remove empty rows and ensure proper formatting
6. **Test with small dataset**: Validate with a small subset before uploading full dataset

## Troubleshooting

### Validation Stuck on "Processing"

- Check browser console for errors
- Try with a smaller dataset first
- Ensure ZIP file is not corrupted

### Warnings About Unmatched IDs

- This is acceptable if <10% of IDs are unmatched
- Review the list of unmatched IDs in the expandable section
- Update CSV or add missing images as needed

### Multiple CSV Files Warning

- The system uses the first CSV file found
- Remove unnecessary CSV files or rename the correct one to be alphabetically first

## Example Valid Dataset

You can create a test dataset with this structure:

```bash
# Create folders
mkdir -p dataset/Images/NORMAL
mkdir -p dataset/Images/PNEUMONIA

# Add some test images (use your actual X-ray images)
cp normal_xray_001.jpg dataset/Images/NORMAL/
cp normal_xray_002.jpg dataset/Images/NORMAL/
cp pneumonia_xray_003.jpg dataset/Images/PNEUMONIA/
cp pneumonia_xray_004.jpg dataset/Images/PNEUMONIA/

# Create CSV
cat > dataset/metadata.csv << EOF
PatientId,Target
normal_xray_001,NORMAL
normal_xray_002,NORMAL
pneumonia_xray_003,PNEUMONIA
pneumonia_xray_004,PNEUMONIA
EOF

# Create ZIP
cd dataset && zip -r ../test_dataset.zip . && cd ..
```

Upload `test_dataset.zip` to validate the implementation.

## Future Enhancements

Potential improvements for future versions:

- Image quality validation
- Duplicate detection
- Class imbalance warnings
- Image dimension consistency checks
- DICOM format support
- Batch validation for multiple datasets
