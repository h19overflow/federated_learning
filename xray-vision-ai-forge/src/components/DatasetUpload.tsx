import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Check, X, Upload, Loader, Info, FileArchive, Image, AlertTriangle, CheckCircle2, FolderOpen, FileText } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';
import InstructionCard from './InstructionCard';
import HelpTooltip from './HelpTooltip';
import JSZip from 'jszip';
import Papa from 'papaparse';
interface DatasetUploadProps {
  onComplete: (data: {
    file: File | null;
    trainSplit: number;
  }) => void;
  initialData?: {
    file: File | null;
    trainSplit: number;
  };
}

interface ValidationStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'success' | 'warning' | 'error';
  message?: string;
  details?: string[];
}

interface ValidationResult {
  isValid: boolean;
  imagesFolder?: string;
  csvFile?: string;
  totalImages: number;
  csvRecords: number;
  matchedImages: number;
  unmatchedPatientIds: string[];
  missingImages: string[];
  targetValues: string[];
  targetColumnName: string;
  warnings: string[];
  errors: string[];
}
const DatasetUpload = ({
  onComplete,
  initialData
}: DatasetUploadProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [trainSplit, setTrainSplit] = useState(80);
  const [datasetSummary, setDatasetSummary] = useState<{
    totalImages: number;
    classes: string[];
  } | null>(null);
  const [validationSteps, setValidationSteps] = useState<ValidationStep[]>([]);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [showValidationDetails, setShowValidationDetails] = useState(false);

  // Initialize with initial data if provided
  useEffect(() => {
    if (initialData) {
      if (initialData.file) {
        setFile(initialData.file);
        setUploadStatus('success');

        // Simulate having dataset summary
        setDatasetSummary({
          totalImages: 5000,
          classes: ['NORMAL', 'PNEUMONIA']
        });
      }
      if (initialData.trainSplit) {
        setTrainSplit(initialData.trainSplit);
      }
    }
  }, [initialData]);
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };
  const validateAndSetFile = (file: File) => {
    // Validate file type (ZIP only)
    if (!file.name.endsWith('.zip')) {
      toast.error('Please upload a ZIP file containing your dataset');
      return;
    }
    setFile(file);
    validateDataset(file);
  };
  const updateValidationStep = (stepId: string, updates: Partial<ValidationStep>) => {
    setValidationSteps(prev => 
      prev.map(step => step.id === stepId ? { ...step, ...updates } : step)
    );
  };

  const validateDataset = async (file: File) => {
    setUploading(true);
    setUploadStatus('uploading');
    setValidationResult(null);
    setShowValidationDetails(true);

    // Initialize validation steps
    const steps: ValidationStep[] = [
      { id: 'zip', name: 'Reading ZIP file', status: 'pending' },
      { id: 'structure', name: 'Checking folder structure', status: 'pending' },
      { id: 'images', name: 'Finding Images directory', status: 'pending' },
      { id: 'csv', name: 'Finding CSV file', status: 'pending' },
      { id: 'csvFormat', name: 'Validating CSV format', status: 'pending' },
      { id: 'matching', name: 'Matching Patient IDs with images', status: 'pending' },
    ];
    setValidationSteps(steps);

    try {
      // Step 1: Read ZIP file
      updateValidationStep('zip', { status: 'processing' });
      const zip = await JSZip.loadAsync(file);
      updateValidationStep('zip', { 
        status: 'success', 
        message: `Loaded ${Object.keys(zip.files).length} files` 
      });

      // Step 2: Check structure
      updateValidationStep('structure', { status: 'processing' });
      const allFiles = Object.keys(zip.files);
      const folders = allFiles.filter(f => zip.files[f].dir);
      updateValidationStep('structure', { 
        status: 'success', 
        message: `Found ${folders.length} directories`,
        details: folders.slice(0, 10)
      });

      // Step 3: Find Images directory (case-insensitive, can be nested)
      updateValidationStep('images', { status: 'processing' });
      let imagesFolder = '';
      const imageFolderPattern = /images\//i;
      
      for (const path of allFiles) {
        if (imageFolderPattern.test(path) && zip.files[path].dir) {
          imagesFolder = path;
          break;
        }
        // Also check if any parent folder is named 'Images'
        const parts = path.split('/');
        const imagesFolderIndex = parts.findIndex(p => p.toLowerCase() === 'images');
        if (imagesFolderIndex !== -1 && !imagesFolder) {
          imagesFolder = parts.slice(0, imagesFolderIndex + 1).join('/') + '/';
          break;
        }
      }

      if (!imagesFolder) {
        updateValidationStep('images', { 
          status: 'error', 
          message: 'Images directory not found in ZIP file' 
        });
        throw new Error('Images directory not found. Please ensure your ZIP contains a folder named "Images"');
      }

      // Get all image files in the Images folder
      const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'];
      const imageFiles = allFiles.filter(path => 
        path.startsWith(imagesFolder) && 
        !zip.files[path].dir &&
        imageExtensions.some(ext => path.toLowerCase().endsWith(ext))
      );

      updateValidationStep('images', { 
        status: 'success', 
        message: `Found ${imageFiles.length} images in "${imagesFolder}"`,
        details: imageFiles.slice(0, 5).map(f => f.replace(imagesFolder, ''))
      });

      // Step 4: Find CSV file (can be nested)
      updateValidationStep('csv', { status: 'processing' });
      let csvFile = '';
      const csvFiles = allFiles.filter(path => 
        path.toLowerCase().endsWith('.csv') && !zip.files[path].dir
      );

      if (csvFiles.length === 0) {
        updateValidationStep('csv', { 
          status: 'error', 
          message: 'No CSV file found in ZIP' 
        });
        throw new Error('CSV file not found. Please include a metadata CSV file in your ZIP');
      }

      if (csvFiles.length > 1) {
        updateValidationStep('csv', { 
          status: 'warning', 
          message: `Found ${csvFiles.length} CSV files, using: ${csvFiles[0]}`,
          details: csvFiles
        });
      }

      csvFile = csvFiles[0];
      updateValidationStep('csv', { 
        status: 'success', 
        message: `Found CSV: ${csvFile}` 
      });

      // Step 5: Validate CSV format
      updateValidationStep('csvFormat', { status: 'processing' });
      const csvContent = await zip.files[csvFile].async('text');
      
      const parseResult = await new Promise<Papa.ParseResult<any>>((resolve) => {
        Papa.parse(csvContent, {
          header: true,
          skipEmptyLines: true,
          complete: resolve
        });
      });

      if (parseResult.errors.length > 0) {
        const criticalErrors = parseResult.errors.filter(e => e.type === 'FieldMismatch');
        if (criticalErrors.length > 0) {
          updateValidationStep('csvFormat', { 
            status: 'warning', 
            message: `CSV has ${criticalErrors.length} parsing warnings`,
            details: criticalErrors.slice(0, 3).map(e => e.message)
          });
        }
      }

      const headers = parseResult.meta.fields || [];
      const normalizedHeaders = headers.map(h => h.toLowerCase().trim());
      
      // Check for patientId column (required)
      const patientIdIndex = normalizedHeaders.indexOf('patientid');
      
      if (patientIdIndex === -1) {
        updateValidationStep('csvFormat', { 
          status: 'error', 
          message: `Missing required column: patientId`,
          details: [`Found columns: ${headers.join(', ')}`]
        });
        throw new Error(`CSV must contain "patientId" column. Found: ${headers.join(', ')}`);
      }

      // Check for target/label column (optional, but recommended)
      // Do NOT use PredictionString as it contains bounding box data, not class labels
      let targetIndex = normalizedHeaders.indexOf('target');
      if (targetIndex === -1) {
        targetIndex = normalizedHeaders.indexOf('label');
      }
      if (targetIndex === -1) {
        targetIndex = normalizedHeaders.indexOf('class');
      }
      if (targetIndex === -1) {
        targetIndex = normalizedHeaders.indexOf('category');
      }

      if (targetIndex === -1) {
        updateValidationStep('csvFormat', { 
          status: 'warning', 
          message: `No target/label column found. Valid CSV with ${parseResult.data.length} records`,
          details: [`Columns: ${headers.join(', ')}`, 'Note: Target column is recommended but not required']
        });
      } else {
        updateValidationStep('csvFormat', { 
          status: 'success', 
          message: `Valid CSV with ${parseResult.data.length} records`,
          details: [`Columns: ${headers.join(', ')}`]
        });
      }

      // Step 6: Match Patient IDs with images
      updateValidationStep('matching', { status: 'processing' });
      
      const csvPatientIds = parseResult.data
        .map((row: any) => {
          const keys = Object.keys(row);
          return row[keys[patientIdIndex]];
        })
        .filter(Boolean)
        .map(id => String(id).trim()); // Ensure string and trim whitespace

      // Extract image filenames without extensions and paths
      // Handle both forward and backslashes, and get just the filename
      const imageFileNames = imageFiles.map(path => {
        const fileName = path.split('/').pop()?.split('\\').pop() || '';
        return fileName.replace(/\.[^/.]+$/, ''); // Remove extension
      });
      
      // Also create a version with full relative paths from Images folder for reference
      const imageFileNamesWithPath = imageFiles.map(path => {
        const relativePath = path.replace(imagesFolder, '');
        return relativePath.replace(/\.[^/.]+$/, '');
      });

      // Create a map of lowercase image names for faster lookup
      const imageNamesLower = imageFileNames.map(name => name.toLowerCase());
      
      // Find matches with improved logic
      const matchedImages: string[] = [];
      const unmatchedPatientIds: string[] = [];
      
      csvPatientIds.forEach(patientId => {
        const patientIdStr = String(patientId).toLowerCase().trim();
        
        // Try multiple matching strategies
        const found = imageNamesLower.some((imgName, index) => {
          const imgLower = imgName.toLowerCase();
          
          // Strategy 1: Exact match
          if (imgLower === patientIdStr) return true;
          
          // Strategy 2: Image name contains patient ID
          if (imgLower.includes(patientIdStr)) return true;
          
          // Strategy 3: Patient ID contains image name (for short IDs)
          if (patientIdStr.includes(imgLower)) return true;
          
          // Strategy 4: Remove common separators and try again
          const cleanPatientId = patientIdStr.replace(/[-_\s]/g, '');
          const cleanImgName = imgLower.replace(/[-_\s]/g, '');
          if (cleanImgName === cleanPatientId) return true;
          if (cleanImgName.includes(cleanPatientId)) return true;
          if (cleanPatientId.includes(cleanImgName)) return true;
          
          return false;
        });
        
        if (found) {
          matchedImages.push(patientId);
        } else {
          unmatchedPatientIds.push(patientId);
        }
      });
      
      // Extensive debugging - log to console
      console.log('=== MATCHING DEBUG INFO ===');
      console.log('Total CSV Patient IDs:', csvPatientIds.length);
      console.log('Total Image Files:', imageFiles.length);
      console.log('\nFirst 10 Patient IDs from CSV:');
      csvPatientIds.slice(0, 10).forEach((id, i) => console.log(`  ${i+1}. "${id}"`));
      console.log('\nFirst 10 Image Files (full ZIP paths):');
      imageFiles.slice(0, 10).forEach((path, i) => console.log(`  ${i+1}. ${path}`));
      console.log('\nFirst 10 Image Names (filename only, no extension):');
      imageFileNames.slice(0, 10).forEach((name, i) => console.log(`  ${i+1}. "${name}"`));
      console.log('\nFirst 10 Image Paths (relative to Images folder):');
      imageFileNamesWithPath.slice(0, 10).forEach((path, i) => console.log(`  ${i+1}. "${path}"`));
      console.log('\n✅ Matched Count:', matchedImages.length);
      console.log('❌ Unmatched Count:', unmatchedPatientIds.length);
      if (unmatchedPatientIds.length > 0 && unmatchedPatientIds.length <= 10) {
        console.log('\nAll Unmatched Patient IDs:');
        unmatchedPatientIds.forEach((id, i) => console.log(`  ${i+1}. "${id}"`));
      } else if (unmatchedPatientIds.length > 10) {
        console.log('\nFirst 20 Unmatched Patient IDs:');
        unmatchedPatientIds.slice(0, 20).forEach((id, i) => console.log(`  ${i+1}. "${id}"`));
      }
      if (matchedImages.length > 0 && matchedImages.length <= 10) {
        console.log('\nAll Matched Patient IDs:');
        matchedImages.forEach((id, i) => console.log(`  ${i+1}. "${id}"`));
      } else if (matchedImages.length > 10) {
        console.log('\nFirst 20 Matched Patient IDs:');
        matchedImages.slice(0, 20).forEach((id, i) => console.log(`  ${i+1}. "${id}"`));
      }
      console.log('========================\n');

      // Get unique values from Target column (if available)
      let targetValues: string[] = [];
      let targetColumnName = '';
      
      if (targetIndex !== -1) {
        // Get the actual column name
        targetColumnName = headers[targetIndex];
        
        // Get all unique values from the target column
        const uniqueValues = [...new Set(
          parseResult.data
            .map((row: any) => {
              const keys = Object.keys(row);
              const value = row[keys[targetIndex]];
              // Convert to string and trim
              return value ? String(value).trim() : '';
            })
            .filter(Boolean) // Remove empty values
        )];
        
        // Limit to first 10 unique values to avoid display issues
        targetValues = uniqueValues.slice(0, 10);
        
        console.log(`Target column "${targetColumnName}" has ${uniqueValues.length} unique values:`, uniqueValues.slice(0, 20));
      }

      const warnings: string[] = [];
      const errors: string[] = [];

      if (unmatchedPatientIds.length > 0) {
        const percentage = (unmatchedPatientIds.length / csvPatientIds.length * 100).toFixed(1);
        warnings.push(`${unmatchedPatientIds.length} patient IDs (${percentage}%) don't have matching images`);
      }

      if (imageFiles.length > csvPatientIds.length) {
        warnings.push(`${imageFiles.length - csvPatientIds.length} images without CSV entries`);
      }

      if (matchedImages.length === 0) {
        // Check if this might be a case where images have different IDs than patient records
        const ratio = imageFiles.length / csvPatientIds.length;
        
        if (ratio > 1.5) {
          // Likely multiple images per patient with different naming
          warnings.push(`Patient IDs don't match image filenames. You may have multiple images per patient.`);
          warnings.push(`Found ${imageFiles.length} images for ${csvPatientIds.length} patient records (${ratio.toFixed(1)}x ratio)`);
          
          const debugDetails = [
            `CSV records: ${csvPatientIds.length}`,
            `Image files: ${imageFiles.length}`,
            `⚠️ No direct ID matches, but continuing (might be multiple images per patient)`,
            `First 3 Patient IDs: ${csvPatientIds.slice(0, 3).join(', ')}`,
            `First 3 Image Names: ${imageFileNames.slice(0, 3).join(', ')}`
          ];
          
          updateValidationStep('matching', { 
            status: 'warning', 
            message: `Patient IDs don't match image names, but ${imageFiles.length} images found`,
            details: debugDetails
          });
          
          // Use image count as matched count for summary purposes
          matchedImages.push(...csvPatientIds);
        } else {
          // This is a real error
          errors.push('No matching images found for any patient ID');
          
          const debugDetails = [
            `CSV records: ${csvPatientIds.length}`,
            `Image files: ${imageFiles.length}`,
            `❌ NO MATCHES FOUND`,
            `First 3 Patient IDs: ${csvPatientIds.slice(0, 3).join(', ')}`,
            `First 3 Image Names: ${imageFileNames.slice(0, 3).join(', ')}`,
            `Check browser console for full debug output`
          ];
          
          updateValidationStep('matching', { 
            status: 'error', 
            message: 'No matches found between CSV and images',
            details: debugDetails
          });
          
          throw new Error('Patient IDs in CSV do not match any image files. Check console for details.');
        }
      }

      const matchPercentage = (matchedImages.length / csvPatientIds.length * 100).toFixed(1);
      
      // Add sample data to details for debugging
      const matchingDetails = [
        `CSV records: ${csvPatientIds.length}`,
        `Image files: ${imageFiles.length}`,
        `Matched: ${matchedImages.length}`,
        `Unmatched IDs: ${unmatchedPatientIds.length}`,
        `Sample Patient IDs: ${csvPatientIds.slice(0, 3).join(', ')}`,
        `Sample Image Names: ${imageFileNames.slice(0, 3).join(', ')}`
      ];
      
      updateValidationStep('matching', { 
        status: unmatchedPatientIds.length > csvPatientIds.length * 0.1 ? 'warning' : 'success', 
        message: `${matchedImages.length} matches found (${matchPercentage}%)`,
        details: matchingDetails
      });

      const result: ValidationResult = {
        isValid: errors.length === 0,
        imagesFolder,
        csvFile,
        totalImages: imageFiles.length,
        csvRecords: csvPatientIds.length,
        matchedImages: matchedImages.length,
        unmatchedPatientIds: unmatchedPatientIds.slice(0, 10),
        missingImages: [],
        targetValues: targetValues,
        targetColumnName: targetColumnName,
        warnings,
        errors
      };

      setValidationResult(result);
      setDatasetSummary({
        totalImages: imageFiles.length, // Use actual image count
        classes: targetValues // Use target values instead of "classes"
      });

      setUploading(false);
      
      if (errors.length === 0) {
        setUploadStatus('success');
        toast.success('Dataset validated successfully!');
      } else {
        setUploadStatus('error');
        toast.error('Dataset validation failed');
      }

    } catch (error) {
      console.error('Validation error:', error);
      setUploading(false);
      setUploadStatus('error');
      toast.error(error instanceof Error ? error.message : 'Failed to validate dataset');
    }
  };
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };
  const handleDragLeave = () => {
    setDragOver(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };
  const handleSubmit = () => {
    onComplete({
      file,
      trainSplit
    });
  };

  // Show a message if we're loading from a saved experiment but don't have the actual file
  const isRestoredExperiment = initialData && initialData.trainSplit && !initialData.file && !file;
  return <div className="animate-fade-in space-y-6">
      {/* Instructions Card */}
      <InstructionCard 
        variant="guide" 
        title="Dataset Requirements"
        className="max-w-4xl mx-auto"
      >
        <div className="space-y-2">
          <p className="font-medium">Your dataset ZIP file must contain:</p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li><strong>Images folder:</strong> A directory named "Images" (case-insensitive, can be nested) containing chest X-ray images</li>
            <li><strong>Metadata CSV:</strong> A CSV file with required column:
              <ul className="list-circle list-inside ml-6 mt-1 text-sm">
                <li><code className="bg-gray-100 px-1 rounded">patientId</code> - Unique identifier matching image filenames (required)</li>
                <li><code className="bg-gray-100 px-1 rounded">Target</code> or <code className="bg-gray-100 px-1 rounded">Label</code> - Class label (optional, e.g., NORMAL, PNEUMONIA)</li>
              </ul>
            </li>
          </ul>
          <p className="mt-3 text-xs">
            <strong>Supported format:</strong> .zip files only. Maximum recommended size: 5GB
          </p>
          <p className="mt-2 text-xs text-gray-600">
            <strong>Note:</strong> The system will automatically validate your dataset structure, CSV format, and attempt to match Patient IDs with image files. If you have multiple images per patient, the system will detect this and proceed with a warning.
          </p>
        </div>
      </InstructionCard>

      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl text-medical-dark flex items-center gap-2">
                <FileArchive className="h-6 w-6" />
                Upload Dataset
              </CardTitle>
              <CardDescription className="mt-2">
                Upload your chest X-ray dataset to begin training
              </CardDescription>
            </div>
            <HelpTooltip
              title="Dataset Format"
              content={
                <div className="space-y-2">
                  <p>Your ZIP file should follow this structure:</p>
                  <pre className="text-xs bg-gray-800 text-gray-100 p-2 rounded mt-2 overflow-x-auto">
{`dataset.zip/
├── Images/
│   ├── patient001.jpeg
│   ├── patient002.jpeg
│   └── ...
└── metadata.csv`}
                  </pre>
                  <p className="mt-2 text-xs">The metadata CSV must include:</p>
                  <ul className="text-xs space-y-1 mt-1">
                    <li>• <strong>patientId</strong> column (required)</li>
                    <li>• <strong>Target</strong> or <strong>Label</strong> column (optional)</li>
                  </ul>
                </div>
              }
              iconClassName="h-5 w-5"
            />
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload Area */}
          <div className={cn("upload-area rounded-lg p-8 text-center cursor-pointer transition-all", dragOver ? "dragover" : "", uploadStatus === 'success' ? "border-status-success" : "")} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} onClick={() => document.getElementById('fileInput')?.click()}>
            <input type="file" id="fileInput" className="hidden" onChange={handleFileChange} accept=".zip" />
            
            {isRestoredExperiment && <div className="space-y-3">
                <Info className="h-12 w-12 mx-auto text-muted-foreground rounded" />
                <p className="text-lg font-medium">Upload a dataset.</p>
                <p className="text-muted-foreground mt-1 text-2xl">No data set Recorded in current session.    upload a new dataset.</p>
                <Button variant="outline" className="mt-2 text-lg bg-teal-700 hover:bg-teal-600 text-slate-50 text-left my-[39px]">Select New  Zip File</Button>
              </div>}
            
            {!isRestoredExperiment && uploadStatus === 'idle' && <div className="space-y-3">
                <Upload className="h-12 w-12 mx-auto text-muted-foreground" />
                <div>
                  <p className="text-lg font-medium">Drop your dataset ZIP file here</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    ZIP file should contain chest X-ray images and a metadata CSV
                  </p>
                </div>
                <Button variant="outline" className="mt-2">
                  Select File
                </Button>
              </div>}
            
            {uploadStatus === 'uploading' && <div className="space-y-3">
                <Loader className="h-12 w-12 mx-auto text-medical animate-spin" />
                <p className="text-lg font-medium">Processing your dataset...</p>
                <div className="w-full bg-muted rounded-full h-2 max-w-xs mx-auto overflow-hidden">
                  <div className="progress-animation h-full rounded-full"></div>
                </div>
              </div>}
            
            {!isRestoredExperiment && uploadStatus === 'success' && <div className="space-y-3">
                <Check className="h-12 w-12 mx-auto text-status-success" />
                <p className="text-lg font-medium">Dataset uploaded successfully</p>
                <p className="text-sm text-muted-foreground">{file?.name}</p>
              </div>}
            
            {uploadStatus === 'error' && <div className="space-y-3">
                <X className="h-12 w-12 mx-auto text-status-error" />
                <p className="text-lg font-medium">Validation failed</p>
                <p className="text-sm text-destructive">{file?.name}</p>
                <Button 
                  variant="outline" 
                  className="mt-2"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (file) validateDataset(file);
                  }}
                >
                  Retry Validation
                </Button>
              </div>}
          </div>
          
          {/* Validation Steps Progress */}
          {validationSteps.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-5 animate-fade-in">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium flex items-center gap-2">
                  <Loader className={cn("h-5 w-5", uploading ? "animate-spin text-medical" : "text-gray-400")} />
                  Validation Progress
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowValidationDetails(!showValidationDetails)}
                  className="text-xs"
                >
                  {showValidationDetails ? 'Hide Details' : 'Show Details'}
                </Button>
              </div>
              
              <div className="space-y-3">
                {validationSteps.map((step) => (
                  <div key={step.id} className="flex items-start gap-3">
                    <div className="mt-0.5">
                      {step.status === 'pending' && (
                        <div className="h-5 w-5 rounded-full border-2 border-gray-300" />
                      )}
                      {step.status === 'processing' && (
                        <Loader className="h-5 w-5 animate-spin text-medical" />
                      )}
                      {step.status === 'success' && (
                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                      )}
                      {step.status === 'warning' && (
                        <AlertTriangle className="h-5 w-5 text-yellow-500" />
                      )}
                      {step.status === 'error' && (
                        <X className="h-5 w-5 text-red-500" />
                      )}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <p className={cn(
                        "text-sm font-medium",
                        step.status === 'success' && "text-green-700",
                        step.status === 'warning' && "text-yellow-700",
                        step.status === 'error' && "text-red-700",
                        step.status === 'processing' && "text-medical",
                        step.status === 'pending' && "text-gray-500"
                      )}>
                        {step.name}
                      </p>
                      
                      {step.message && (
                        <p className={cn(
                          "text-xs mt-1",
                          step.status === 'error' ? "text-red-600" : 
                          step.status === 'warning' ? "text-yellow-600" : 
                          "text-gray-600"
                        )}>
                          {step.message}
                        </p>
                      )}
                      
                      {showValidationDetails && step.details && step.details.length > 0 && (
                        <ul className="mt-2 text-xs text-gray-500 space-y-1 pl-3">
                          {step.details.map((detail, idx) => (
                            <li key={idx} className="flex items-start gap-1">
                              <span className="text-gray-400">•</span>
                              <span className="break-all">{detail}</span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Validation Results Summary */}
          {validationResult && (
            <div className={cn(
              "border rounded-lg p-5 animate-fade-in",
              validationResult.isValid ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"
            )}>
              <div className="flex items-start gap-3">
                {validationResult.isValid ? (
                  <CheckCircle2 className="h-6 w-6 text-green-600 mt-0.5" />
                ) : (
                  <X className="h-6 w-6 text-red-600 mt-0.5" />
                )}
                
                <div className="flex-1">
                  <h3 className={cn(
                    "text-lg font-semibold mb-2",
                    validationResult.isValid ? "text-green-800" : "text-red-800"
                  )}>
                    {validationResult.isValid ? 'Dataset Validation Successful' : 'Dataset Validation Failed'}
                  </h3>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="bg-white/50 p-3 rounded-md">
                      <p className="text-xs text-gray-600">Images Found</p>
                      <p className="text-xl font-semibold text-gray-900">{validationResult.totalImages}</p>
                    </div>
                    <div className="bg-white/50 p-3 rounded-md">
                      <p className="text-xs text-gray-600">CSV Records</p>
                      <p className="text-xl font-semibold text-gray-900">{validationResult.csvRecords}</p>
                    </div>
                    <div className="bg-white/50 p-3 rounded-md">
                      <p className="text-xs text-gray-600">Matched</p>
                      <p className="text-xl font-semibold text-green-600">{validationResult.matchedImages}</p>
                    </div>
                    <div className="bg-white/50 p-3 rounded-md">
                      <p className="text-xs text-gray-600">
                        {validationResult.targetColumnName ? `${validationResult.targetColumnName} Values` : 'Target Values'}
                      </p>
                      <p className="text-xl font-semibold text-gray-900">
                        {validationResult.targetValues.length > 0 ? validationResult.targetValues.length : 'N/A'}
                      </p>
                    </div>
                  </div>
                  
                  {validationResult.warnings.length > 0 && (
                    <div className="mb-3">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className="h-4 w-4 text-yellow-600" />
                        <p className="text-sm font-medium text-yellow-800">Warnings:</p>
                      </div>
                      <ul className="text-sm text-yellow-700 space-y-1 pl-6">
                        {validationResult.warnings.map((warning, idx) => (
                          <li key={idx} className="list-disc">{warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {validationResult.errors.length > 0 && (
                    <div className="mb-3">
                      <div className="flex items-center gap-2 mb-2">
                        <X className="h-4 w-4 text-red-600" />
                        <p className="text-sm font-medium text-red-800">Errors:</p>
                      </div>
                      <ul className="text-sm text-red-700 space-y-1 pl-6">
                        {validationResult.errors.map((error, idx) => (
                          <li key={idx} className="list-disc">{error}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {validationResult.unmatchedPatientIds.length > 0 && (
                    <details className="mt-3">
                      <summary className="text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900">
                        View unmatched Patient IDs ({validationResult.unmatchedPatientIds.length})
                      </summary>
                      <div className="mt-2 p-3 bg-white/50 rounded text-xs text-gray-600 max-h-32 overflow-y-auto">
                        {validationResult.unmatchedPatientIds.join(', ')}
                      </div>
                    </details>
                  )}
                  
                  <div className="mt-4 pt-4 border-t border-gray-300">
                      <div className="flex items-start gap-2">
                        <FolderOpen className="h-4 w-4 text-gray-500 mt-0.5" />
                        <div className="text-xs text-gray-600">
                          <p><strong>Images folder:</strong> {validationResult.imagesFolder}</p>
                          <p className="mt-1"><strong>CSV file:</strong> {validationResult.csvFile}</p>
                          {validationResult.targetColumnName && (
                            <p className="mt-1">
                              <strong>{validationResult.targetColumnName} unique values ({validationResult.targetValues.length}):</strong>{' '}
                              {validationResult.targetValues.length > 0 
                                ? validationResult.targetValues.length <= 10
                                  ? validationResult.targetValues.join(', ')
                                  : `${validationResult.targetValues.slice(0, 10).join(', ')} ... (showing first 10)`
                                : 'None'}
                            </p>
                          )}
                          {!validationResult.targetColumnName && (
                            <p className="mt-1">
                              <strong>Target column:</strong> Not found (Target, Label, Class, or Category column missing)
                            </p>
                          )}
                        </div>
                      </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Dataset Summary (shown only after successful upload) */}
          {(datasetSummary || isRestoredExperiment) && <div className="bg-muted/50 p-4 rounded-lg animate-fade-in">
              <h3 className="text-lg font-medium flex items-center gap-2">
                <Info className="h-5 w-5 text-medical" />
                Dataset Summary
              </h3>
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white p-3 rounded-md shadow-sm">
                  <p className="text-sm text-muted-foreground">Total Images</p>
                  <p className="text-2xl font-semibold text-medical-dark">{datasetSummary?.totalImages || 5000}</p>
                </div>
                <div className="bg-white p-3 rounded-md shadow-sm">
                  <p className="text-sm text-muted-foreground">Target Values</p>
                  <p className="text-2xl font-semibold text-medical-dark">
                    {datasetSummary?.classes && datasetSummary.classes.length > 0 
                      ? datasetSummary.classes.length 
                      : 'N/A'
                    }
                    {datasetSummary?.classes && datasetSummary.classes.length > 0 && datasetSummary.classes.length <= 5 && (
                      <span className="text-sm font-normal text-muted-foreground ml-2 block mt-1">
                        ({datasetSummary.classes.join(', ')})
                      </span>
                    )}
                    {datasetSummary?.classes && datasetSummary.classes.length > 5 && (
                      <span className="text-xs font-normal text-muted-foreground ml-2 block mt-1">
                        (First 5: {datasetSummary.classes.slice(0, 5).join(', ')})
                      </span>
                    )}
                    {(!datasetSummary?.classes || datasetSummary.classes.length === 0) && (
                      <span className="text-xs font-normal text-muted-foreground ml-2 block mt-1">
                        No target column found
                      </span>
                    )}
                  </p>
                </div>
              </div>
            </div>}
          
          {/* Train/Validation Split Slider */}
          <div className="bg-gray-50 p-5 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium flex items-center gap-2">
                <Image className="h-5 w-5 text-medical" />
                Train/Validation Split
              </h3>
              <HelpTooltip
                content={
                  <div className="space-y-2">
                    <p><strong>Training Set:</strong> Used to train the model's weights</p>
                    <p><strong>Validation Set:</strong> Used to evaluate model performance during training</p>
                    <p className="text-xs mt-2 pt-2 border-t">
                      Recommended: 80/20 split. Higher training percentage gives more data for learning,
                      but less data for validation.
                    </p>
                  </div>
                }
              />
            </div>
            <div className="flex flex-col space-y-3">
              <div className="flex justify-between text-sm font-medium">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                  <span>Training: {trainSplit}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span>Validation: {100 - trainSplit}%</span>
                </div>
              </div>
              <Slider 
                value={[trainSplit]} 
                min={50} 
                max={90} 
                step={5} 
                className="w-full" 
                onValueChange={values => setTrainSplit(values[0])} 
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>More validation</span>
                <span>More training</span>
              </div>
            </div>
          </div>

          {/* Helpful tip */}
          {(datasetSummary || isRestoredExperiment) && (
            <InstructionCard variant="tip" className="animate-fade-in">
              <p>
                <strong>Pro tip:</strong> Start with the default 80/20 split. 
                If your dataset is small ({datasetSummary && datasetSummary.totalImages < 500 && 'like this one'}), 
                consider using 70/30 to have more validation data for better performance assessment.
              </p>
            </InstructionCard>
          )}
        </CardContent>
        <CardFooter>
          <Button onClick={handleSubmit} disabled={!isRestoredExperiment && uploadStatus !== 'success'} className="ml-auto bg-medical hover:bg-medical-dark text-lg">
            Continue
          </Button>
        </CardFooter>
      </Card>
    </div>;
};
export default DatasetUpload;