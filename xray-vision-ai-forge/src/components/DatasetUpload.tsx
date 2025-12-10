import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Check, X, Upload, Loader, Info, FileArchive, Image, AlertTriangle, CheckCircle2, FolderOpen, FileText, ChevronDown, ChevronUp } from 'lucide-react';
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

      // Step 3: Find Images directory
      updateValidationStep('images', { status: 'processing' });
      let imagesFolder = '';
      const imageFolderPattern = /images\//i;

      for (const path of allFiles) {
        if (imageFolderPattern.test(path) && zip.files[path].dir) {
          imagesFolder = path;
          break;
        }
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

      // Step 4: Find CSV file
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

      const patientIdIndex = normalizedHeaders.indexOf('patientid');

      if (patientIdIndex === -1) {
        updateValidationStep('csvFormat', {
          status: 'error',
          message: `Missing required column: patientId`,
          details: [`Found columns: ${headers.join(', ')}`]
        });
        throw new Error(`CSV must contain "patientId" column. Found: ${headers.join(', ')}`);
      }

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
        .map(id => String(id).trim());

      const imageFileNames = imageFiles.map(path => {
        const fileName = path.split('/').pop()?.split('\\').pop() || '';
        return fileName.replace(/\.[^/.]+$/, '');
      });

      const imageFileNamesWithPath = imageFiles.map(path => {
        const relativePath = path.replace(imagesFolder, '');
        return relativePath.replace(/\.[^/.]+$/, '');
      });

      const imageNamesLower = imageFileNames.map(name => name.toLowerCase());

      const matchedImages: string[] = [];
      const unmatchedPatientIds: string[] = [];

      csvPatientIds.forEach(patientId => {
        const patientIdStr = String(patientId).toLowerCase().trim();

        const found = imageNamesLower.some((imgName, index) => {
          const imgLower = imgName.toLowerCase();

          if (imgLower === patientIdStr) return true;
          if (imgLower.includes(patientIdStr)) return true;
          if (patientIdStr.includes(imgLower)) return true;

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

      // Get unique values from Target column
      let targetValues: string[] = [];
      let targetColumnName = '';

      if (targetIndex !== -1) {
        targetColumnName = headers[targetIndex];

        const uniqueValues = [...new Set(
          parseResult.data
            .map((row: any) => {
              const keys = Object.keys(row);
              const value = row[keys[targetIndex]];
              return value ? String(value).trim() : '';
            })
            .filter(Boolean)
        )];

        targetValues = uniqueValues.slice(0, 10);
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
        const ratio = imageFiles.length / csvPatientIds.length;

        if (ratio > 1.5) {
          warnings.push(`Patient IDs don't match image filenames. You may have multiple images per patient.`);
          warnings.push(`Found ${imageFiles.length} images for ${csvPatientIds.length} patient records (${ratio.toFixed(1)}x ratio)`);

          const debugDetails = [
            `CSV records: ${csvPatientIds.length}`,
            `Image files: ${imageFiles.length}`,
            `First 3 Patient IDs: ${csvPatientIds.slice(0, 3).join(', ')}`,
            `First 3 Image Names: ${imageFileNames.slice(0, 3).join(', ')}`
          ];

          updateValidationStep('matching', {
            status: 'warning',
            message: `Patient IDs don't match image names, but ${imageFiles.length} images found`,
            details: debugDetails
          });

          matchedImages.push(...csvPatientIds);
        } else {
          errors.push('No matching images found for any patient ID');

          const debugDetails = [
            `CSV records: ${csvPatientIds.length}`,
            `Image files: ${imageFiles.length}`,
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
        totalImages: imageFiles.length,
        classes: targetValues
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

  const isRestoredExperiment = initialData && initialData.trainSplit && !initialData.file && !file;

  return (
    <div className="space-y-8" style={{ animation: 'fadeIn 0.5s ease-out' }}>
      {/* Instructions Card */}
      <InstructionCard
        variant="guide"
        title="Dataset Requirements"
        className="max-w-4xl mx-auto"
      >
        <div className="space-y-3">
          <p className="font-medium text-[hsl(172_43%_20%)]">Your dataset ZIP file must contain:</p>
          <ul className="space-y-2 ml-1">
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_28%)] mt-2 flex-shrink-0" />
              <span><strong className="text-[hsl(172_43%_20%)]">Images folder:</strong> A directory named "Images" containing chest X-ray images</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_28%)] mt-2 flex-shrink-0" />
              <div>
                <strong className="text-[hsl(172_43%_20%)]">Metadata CSV:</strong> A CSV file with columns:
                <div className="flex flex-wrap gap-2 mt-2">
                  <code className="px-2 py-1 rounded-lg bg-[hsl(168_25%_94%)] text-[hsl(172_43%_20%)] text-xs font-medium">patientId</code>
                  <span className="text-[hsl(215_15%_50%)] text-xs">required</span>
                  <code className="px-2 py-1 rounded-lg bg-[hsl(168_25%_94%)] text-[hsl(172_43%_20%)] text-xs font-medium">Target / Label</code>
                  <span className="text-[hsl(215_15%_50%)] text-xs">optional</span>
                </div>
              </div>
            </li>
          </ul>
          <p className="text-xs text-[hsl(215_15%_50%)] pt-2 border-t border-[hsl(168_20%_90%)]">
            <strong>Supported format:</strong> .zip files only | <strong>Maximum recommended:</strong> 5GB
          </p>
        </div>
      </InstructionCard>

      {/* Main Upload Card */}
      <div className="w-full max-w-4xl mx-auto">
        <div className="bg-white rounded-[2rem] border border-[hsl(210_15%_92%)] shadow-lg shadow-[hsl(172_40%_85%)]/20 overflow-hidden">
          {/* Header */}
          <div className="px-8 py-6 border-b border-[hsl(210_15%_94%)] bg-gradient-to-r from-[hsl(168_25%_98%)] to-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-2xl bg-[hsl(172_40%_94%)]">
                  <FileArchive className="h-6 w-6 text-[hsl(172_63%_28%)]" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-[hsl(172_43%_15%)] tracking-tight">
                    Upload Dataset
                  </h2>
                  <p className="text-[hsl(215_15%_50%)] mt-1">
                    Upload your chest X-ray dataset to begin training
                  </p>
                </div>
              </div>
              <HelpTooltip
                title="Dataset Format"
                content={
                  <div className="space-y-3">
                    <p className="text-[hsl(172_43%_20%)]">Your ZIP file should follow this structure:</p>
                    <pre className="text-xs bg-[hsl(172_30%_12%)] text-[hsl(168_15%_90%)] p-3 rounded-xl overflow-x-auto font-mono">
{`dataset.zip/
├── Images/
│   ├── patient001.jpeg
│   ├── patient002.jpeg
│   └── ...
└── metadata.csv`}
                    </pre>
                  </div>
                }
                iconClassName="h-5 w-5"
              />
            </div>
          </div>

          {/* Content */}
          <div className="p-8 space-y-8">
            {/* Upload Area */}
            <div
              className={cn(
                "relative rounded-2xl border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-300",
                dragOver
                  ? "border-[hsl(172_63%_35%)] bg-[hsl(168_40%_97%)] scale-[1.01]"
                  : uploadStatus === 'success'
                    ? "border-[hsl(172_63%_35%)] bg-[hsl(168_35%_97%)]"
                    : uploadStatus === 'error'
                      ? "border-[hsl(0_72%_51%)] bg-[hsl(0_60%_98%)]"
                      : "border-[hsl(210_15%_88%)] bg-[hsl(168_25%_98%)] hover:border-[hsl(172_40%_70%)] hover:bg-[hsl(168_30%_97%)]"
              )}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById('fileInput')?.click()}
            >
              <input type="file" id="fileInput" className="hidden" onChange={handleFileChange} accept=".zip" />

              {isRestoredExperiment && (
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-[hsl(168_25%_94%)] flex items-center justify-center">
                    <Info className="h-8 w-8 text-[hsl(215_15%_55%)]" />
                  </div>
                  <p className="text-xl font-semibold text-[hsl(172_43%_15%)]">Upload a dataset</p>
                  <p className="text-[hsl(215_15%_50%)]">No dataset recorded in current session</p>
                  <Button className="mt-4 bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-xl px-6 py-2.5 shadow-md shadow-[hsl(172_63%_22%)]/20">
                    Select ZIP File
                  </Button>
                </div>
              )}

              {!isRestoredExperiment && uploadStatus === 'idle' && (
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
                    <Upload className="h-8 w-8 text-[hsl(172_63%_35%)]" />
                  </div>
                  <div>
                    <p className="text-xl font-semibold text-[hsl(172_43%_15%)]">Drop your dataset ZIP file here</p>
                    <p className="text-[hsl(215_15%_50%)] mt-2">
                      ZIP file should contain chest X-ray images and a metadata CSV
                    </p>
                  </div>
                  <Button variant="outline" className="mt-2 rounded-xl border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)]">
                    Select File
                  </Button>
                </div>
              )}

              {uploadStatus === 'uploading' && (
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
                    <Loader className="h-8 w-8 text-[hsl(172_63%_35%)] animate-spin" />
                  </div>
                  <p className="text-xl font-semibold text-[hsl(172_43%_15%)]">Processing your dataset...</p>
                  <div className="w-full bg-[hsl(168_25%_92%)] rounded-full h-2 max-w-xs mx-auto overflow-hidden">
                    <div className="h-full rounded-full bg-gradient-to-r from-[hsl(172_63%_35%)] to-[hsl(172_63%_28%)] animate-pulse" style={{ width: '60%' }} />
                  </div>
                </div>
              )}

              {!isRestoredExperiment && uploadStatus === 'success' && (
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-[hsl(172_50%_92%)] flex items-center justify-center">
                    <Check className="h-8 w-8 text-[hsl(172_63%_28%)]" />
                  </div>
                  <p className="text-xl font-semibold text-[hsl(172_43%_15%)]">Dataset uploaded successfully</p>
                  <p className="text-[hsl(215_15%_50%)]">{file?.name}</p>
                </div>
              )}

              {uploadStatus === 'error' && (
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-[hsl(0_60%_95%)] flex items-center justify-center">
                    <X className="h-8 w-8 text-[hsl(0_72%_51%)]" />
                  </div>
                  <p className="text-xl font-semibold text-[hsl(172_43%_15%)]">Validation failed</p>
                  <p className="text-[hsl(0_72%_45%)]">{file?.name}</p>
                  <Button
                    variant="outline"
                    className="mt-2 rounded-xl border-[hsl(0_50%_80%)] text-[hsl(0_72%_45%)] hover:bg-[hsl(0_50%_97%)]"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (file) validateDataset(file);
                    }}
                  >
                    Retry Validation
                  </Button>
                </div>
              )}
            </div>

            {/* Validation Steps Progress */}
            {validationSteps.length > 0 && (
              <div className="bg-[hsl(168_25%_98%)] border border-[hsl(168_20%_92%)] rounded-2xl p-6" style={{ animation: 'fadeIn 0.3s ease-out' }}>
                <div className="flex items-center justify-between mb-5">
                  <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3">
                    <div className={cn(
                      "p-2 rounded-xl",
                      uploading ? "bg-[hsl(172_40%_94%)]" : "bg-[hsl(168_20%_94%)]"
                    )}>
                      <Loader className={cn(
                        "h-5 w-5",
                        uploading ? "animate-spin text-[hsl(172_63%_35%)]" : "text-[hsl(215_15%_55%)]"
                      )} />
                    </div>
                    Validation Progress
                  </h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowValidationDetails(!showValidationDetails)}
                    className="text-xs text-[hsl(172_43%_30%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(168_25%_94%)] rounded-lg"
                  >
                    {showValidationDetails ? (
                      <>Hide Details <ChevronUp className="ml-1 h-4 w-4" /></>
                    ) : (
                      <>Show Details <ChevronDown className="ml-1 h-4 w-4" /></>
                    )}
                  </Button>
                </div>

                <div className="space-y-3">
                  {validationSteps.map((step, index) => (
                    <div
                      key={step.id}
                      className="flex items-start gap-4"
                      style={{
                        animation: 'fadeIn 0.3s ease-out',
                        animationDelay: `${index * 0.05}s`
                      }}
                    >
                      <div className="mt-0.5">
                        {step.status === 'pending' && (
                          <div className="h-6 w-6 rounded-full border-2 border-[hsl(210_15%_85%)] bg-white" />
                        )}
                        {step.status === 'processing' && (
                          <div className="h-6 w-6 rounded-full bg-[hsl(172_40%_94%)] flex items-center justify-center">
                            <Loader className="h-4 w-4 animate-spin text-[hsl(172_63%_35%)]" />
                          </div>
                        )}
                        {step.status === 'success' && (
                          <div className="h-6 w-6 rounded-full bg-[hsl(172_50%_92%)] flex items-center justify-center">
                            <CheckCircle2 className="h-4 w-4 text-[hsl(172_63%_28%)]" />
                          </div>
                        )}
                        {step.status === 'warning' && (
                          <div className="h-6 w-6 rounded-full bg-[hsl(35_60%_92%)] flex items-center justify-center">
                            <AlertTriangle className="h-4 w-4 text-[hsl(35_70%_45%)]" />
                          </div>
                        )}
                        {step.status === 'error' && (
                          <div className="h-6 w-6 rounded-full bg-[hsl(0_60%_95%)] flex items-center justify-center">
                            <X className="h-4 w-4 text-[hsl(0_72%_51%)]" />
                          </div>
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <p className={cn(
                          "text-sm font-medium",
                          step.status === 'success' && "text-[hsl(172_43%_25%)]",
                          step.status === 'warning' && "text-[hsl(35_70%_40%)]",
                          step.status === 'error' && "text-[hsl(0_72%_45%)]",
                          step.status === 'processing' && "text-[hsl(172_63%_28%)]",
                          step.status === 'pending' && "text-[hsl(215_15%_55%)]"
                        )}>
                          {step.name}
                        </p>

                        {step.message && (
                          <p className={cn(
                            "text-xs mt-1",
                            step.status === 'error' ? "text-[hsl(0_60%_50%)]" :
                            step.status === 'warning' ? "text-[hsl(35_60%_45%)]" :
                            "text-[hsl(215_15%_55%)]"
                          )}>
                            {step.message}
                          </p>
                        )}

                        {showValidationDetails && step.details && step.details.length > 0 && (
                          <ul className="mt-2 text-xs text-[hsl(215_15%_55%)] space-y-1 pl-3 border-l-2 border-[hsl(168_20%_90%)]">
                            {step.details.map((detail, idx) => (
                              <li key={idx} className="break-all">
                                {detail}
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
              <div
                className={cn(
                  "rounded-2xl p-6 border",
                  validationResult.isValid
                    ? "bg-[hsl(172_40%_97%)] border-[hsl(172_40%_85%)]"
                    : "bg-[hsl(0_50%_98%)] border-[hsl(0_50%_85%)]"
                )}
                style={{ animation: 'fadeIn 0.4s ease-out' }}
              >
                <div className="flex items-start gap-4">
                  {validationResult.isValid ? (
                    <div className="p-3 rounded-xl bg-[hsl(172_50%_92%)]">
                      <CheckCircle2 className="h-6 w-6 text-[hsl(172_63%_28%)]" />
                    </div>
                  ) : (
                    <div className="p-3 rounded-xl bg-[hsl(0_60%_95%)]">
                      <X className="h-6 w-6 text-[hsl(0_72%_51%)]" />
                    </div>
                  )}

                  <div className="flex-1">
                    <h3 className={cn(
                      "text-xl font-semibold mb-4",
                      validationResult.isValid ? "text-[hsl(172_43%_20%)]" : "text-[hsl(0_72%_40%)]"
                    )}>
                      {validationResult.isValid ? 'Dataset Validation Successful' : 'Dataset Validation Failed'}
                    </h3>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
                      <div className="bg-white/60 backdrop-blur p-4 rounded-xl border border-[hsl(168_20%_92%)]">
                        <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide">Images Found</p>
                        <p className="text-2xl font-semibold text-[hsl(172_43%_20%)] mt-1">{validationResult.totalImages}</p>
                      </div>
                      <div className="bg-white/60 backdrop-blur p-4 rounded-xl border border-[hsl(168_20%_92%)]">
                        <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide">CSV Records</p>
                        <p className="text-2xl font-semibold text-[hsl(172_43%_20%)] mt-1">{validationResult.csvRecords}</p>
                      </div>
                      <div className="bg-white/60 backdrop-blur p-4 rounded-xl border border-[hsl(168_20%_92%)]">
                        <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide">Matched</p>
                        <p className="text-2xl font-semibold text-[hsl(172_63%_28%)] mt-1">{validationResult.matchedImages}</p>
                      </div>
                      <div className="bg-white/60 backdrop-blur p-4 rounded-xl border border-[hsl(168_20%_92%)]">
                        <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide">
                          {validationResult.targetColumnName || 'Target Values'}
                        </p>
                        <p className="text-2xl font-semibold text-[hsl(172_43%_20%)] mt-1">
                          {validationResult.targetValues.length > 0 ? validationResult.targetValues.length : 'N/A'}
                        </p>
                      </div>
                    </div>

                    {validationResult.warnings.length > 0 && (
                      <div className="mb-4 p-4 rounded-xl bg-[hsl(35_60%_96%)] border border-[hsl(35_50%_85%)]">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertTriangle className="h-4 w-4 text-[hsl(35_70%_45%)]" />
                          <p className="text-sm font-medium text-[hsl(35_70%_35%)]">Warnings</p>
                        </div>
                        <ul className="text-sm text-[hsl(35_60%_40%)] space-y-1">
                          {validationResult.warnings.map((warning, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(35_70%_50%)] mt-2 flex-shrink-0" />
                              {warning}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {validationResult.errors.length > 0 && (
                      <div className="mb-4 p-4 rounded-xl bg-[hsl(0_60%_97%)] border border-[hsl(0_50%_85%)]">
                        <div className="flex items-center gap-2 mb-2">
                          <X className="h-4 w-4 text-[hsl(0_72%_51%)]" />
                          <p className="text-sm font-medium text-[hsl(0_72%_40%)]">Errors</p>
                        </div>
                        <ul className="text-sm text-[hsl(0_60%_45%)] space-y-1">
                          {validationResult.errors.map((error, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(0_72%_51%)] mt-2 flex-shrink-0" />
                              {error}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {validationResult.unmatchedPatientIds.length > 0 && (
                      <details className="mt-4">
                        <summary className="text-sm font-medium text-[hsl(215_15%_45%)] cursor-pointer hover:text-[hsl(172_43%_25%)] transition-colors">
                          View unmatched Patient IDs ({validationResult.unmatchedPatientIds.length})
                        </summary>
                        <div className="mt-2 p-3 bg-white/60 rounded-xl text-xs text-[hsl(215_15%_55%)] max-h-32 overflow-y-auto font-mono">
                          {validationResult.unmatchedPatientIds.join(', ')}
                        </div>
                      </details>
                    )}

                    {/* File info */}
                    <div className="mt-5 pt-5 border-t border-[hsl(168_20%_90%)]">
                      <div className="flex items-start gap-3">
                        <FolderOpen className="h-4 w-4 text-[hsl(215_15%_55%)] mt-0.5" />
                        <div className="text-xs text-[hsl(215_15%_55%)] space-y-1">
                          <p><strong>Images folder:</strong> {validationResult.imagesFolder}</p>
                          <p><strong>CSV file:</strong> {validationResult.csvFile}</p>
                          {validationResult.targetColumnName && (
                            <p>
                              <strong>{validationResult.targetColumnName} values:</strong>{' '}
                              {validationResult.targetValues.length <= 10
                                ? validationResult.targetValues.join(', ')
                                : `${validationResult.targetValues.slice(0, 10).join(', ')} ...`}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Dataset Summary */}
            {(datasetSummary || isRestoredExperiment) && (
              <div
                className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]"
                style={{ animation: 'fadeIn 0.4s ease-out' }}
              >
                <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-4">
                  <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                    <Info className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                  </div>
                  Dataset Summary
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-white p-5 rounded-xl border border-[hsl(210_15%_92%)] shadow-sm">
                    <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide">Total Images</p>
                    <p className="text-3xl font-semibold text-[hsl(172_43%_20%)] mt-1">{datasetSummary?.totalImages || 5000}</p>
                  </div>
                  <div className="bg-white p-5 rounded-xl border border-[hsl(210_15%_92%)] shadow-sm">
                    <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide">Target Values</p>
                    <p className="text-3xl font-semibold text-[hsl(172_43%_20%)] mt-1">
                      {datasetSummary?.classes && datasetSummary.classes.length > 0
                        ? datasetSummary.classes.length
                        : 'N/A'
                      }
                    </p>
                    {datasetSummary?.classes && datasetSummary.classes.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-2">
                        {datasetSummary.classes.slice(0, 5).map((cls, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 rounded-lg bg-[hsl(168_25%_94%)] text-[hsl(172_43%_25%)] text-xs font-medium"
                          >
                            {cls}
                          </span>
                        ))}
                        {datasetSummary.classes.length > 5 && (
                          <span className="px-2 py-1 text-[hsl(215_15%_55%)] text-xs">
                            +{datasetSummary.classes.length - 5} more
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Train/Validation Split Slider */}
            <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3">
                  <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                    <Image className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                  </div>
                  Train/Validation Split
                </h3>
                <HelpTooltip
                  content={
                    <div className="space-y-2">
                      <p><strong className="text-[hsl(172_43%_20%)]">Training Set:</strong> Used to train the model's weights</p>
                      <p><strong className="text-[hsl(172_43%_20%)]">Validation Set:</strong> Used to evaluate model performance during training</p>
                      <p className="text-xs mt-2 pt-2 border-t border-[hsl(168_20%_90%)]">
                        Recommended: 80/20 split for most datasets.
                      </p>
                    </div>
                  }
                />
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full bg-[hsl(172_63%_35%)]" />
                    <span className="text-sm font-medium text-[hsl(172_43%_20%)]">Training: {trainSplit}%</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full bg-[hsl(210_60%_50%)]" />
                    <span className="text-sm font-medium text-[hsl(172_43%_20%)]">Validation: {100 - trainSplit}%</span>
                  </div>
                </div>

                {/* Custom styled slider track */}
                <div className="relative py-2">
                  <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-2 rounded-full overflow-hidden bg-[hsl(210_40%_85%)]">
                    <div
                      className="h-full bg-gradient-to-r from-[hsl(172_63%_35%)] to-[hsl(172_63%_28%)] rounded-full"
                      style={{ width: `${trainSplit}%` }}
                    />
                  </div>
                  <Slider
                    value={[trainSplit]}
                    min={50}
                    max={90}
                    step={5}
                    className="relative z-10 [&_[role=slider]]:bg-white [&_[role=slider]]:border-2 [&_[role=slider]]:border-[hsl(172_63%_35%)] [&_[role=slider]]:shadow-md [&_[role=slider]]:w-5 [&_[role=slider]]:h-5"
                    onValueChange={values => setTrainSplit(values[0])}
                  />
                </div>

                <div className="flex justify-between text-xs text-[hsl(215_15%_55%)]">
                  <span>More validation data</span>
                  <span>More training data</span>
                </div>
              </div>
            </div>

            {/* Pro tip */}
            {(datasetSummary || isRestoredExperiment) && (
              <InstructionCard variant="tip" className="animate-fade-in">
                <p>
                  <strong>Pro tip:</strong> Start with the default 80/20 split.
                  {datasetSummary && datasetSummary.totalImages < 500 && (
                    <> Since your dataset is smaller, consider using 70/30 for better validation accuracy.</>
                  )}
                </p>
              </InstructionCard>
            )}
          </div>

          {/* Footer */}
          <div className="px-8 py-6 border-t border-[hsl(210_15%_94%)] bg-[hsl(168_25%_99%)]">
            <Button
              onClick={handleSubmit}
              disabled={!isRestoredExperiment && uploadStatus !== 'success'}
              className="ml-auto flex bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white text-base px-8 py-6 rounded-xl shadow-lg shadow-[hsl(172_63%_22%)]/20 transition-all duration-300 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/30 hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-lg"
            >
              Continue
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetUpload;
