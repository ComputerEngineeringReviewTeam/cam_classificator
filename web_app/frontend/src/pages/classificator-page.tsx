// src/pages/ClassificatorPage.tsx
import React, { useState, useRef, useEffect } from 'react';
import api from '../services/api';
import { ClassificationResult } from '../models/classificator-responses';
import { AxiosError } from "axios";
import ImageDisplay from '../components/classificator/image-display';

const API_URL = '/classificator';
const ACCEPTED_FILES = "image/png, image/jpeg, image/jpg";


const ClassificatorPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<ClassificationResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up object URL when component unmounts or image changes
  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Reset state on new file selection
    setAnalysisResult(null);
    setError(null);
    setSelectedFile(null);
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
      setImageUrl(null);
    }

    if (file) {
      if (!ACCEPTED_FILES.includes(file.type)) {
        setError(`File type not allowed. Please select ${ACCEPTED_FILES}.`);
        return;
      }
      setSelectedFile(file);
      setImageUrl(URL.createObjectURL(file));
    }

    // Clear the input value so the same file can be selected again if needed
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await api.post<ClassificationResult>(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        }
      });
      setAnalysisResult(response.data);
    } catch (err) {
      const axiosError = err as AxiosError<{ message?: string }>;
      console.error("Upload failed:", err);
      setError(axiosError.response?.data?.message || axiosError.message || 'Analysis failed. Check console and network tab.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto max-w-full min-w-full mt-6 mb-10 px-4">
      {/* --- Header & Controls --- */}
      <div className="max-w-4xl mx-auto bg-white p-6 md:p-8 rounded-xl shadow-2xl shadow-blue-300 mb-8">
        <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-800 mb-6">Image Classificator</h2>

        {error && <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded"
                       role="alert">{error}</div>}

        <div className="flex flex-col sm:flex-row gap-4 items-center justify-center">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept={ACCEPTED_FILES}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="bg-gray-600 hover:bg-gray-700 text-white font-semibold py-2 px-6 rounded-lg shadow disabled:opacity-50 w-full sm:w-auto"
          >
            {selectedFile ? `File: ${selectedFile.name}` : 'Select Image'}
          </button>

          <button
            onClick={handleUpload}
            disabled={!selectedFile || isLoading}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-6 rounded-lg shadow disabled:opacity-50 w-full sm:w-auto"
          >
            {isLoading ? 'Analyzing...' : 'Analyze Image'}
          </button>
        </div>
        {isLoading && (
          <div className="flex justify-center mt-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-700"></div>
          </div>
        )}
      </div>

      {/* --- Image & Overlay Display --- */}
      {/* Only show if image exists AND we are not loading AND no error occurred during load */}
      {(imageUrl && !isLoading && (!error || analysisResult)) && (
        <ImageDisplay
          imageUrl={imageUrl}
          analysisResult={analysisResult}
        />
      )}
    </div>
  );
};

export default ClassificatorPage;