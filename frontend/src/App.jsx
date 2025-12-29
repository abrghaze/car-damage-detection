import React, { useState, useCallback } from 'react';
import { Upload, Camera, AlertTriangle, CheckCircle, Loader2, Car, Eye, Layers, BarChart3, Clock, X } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

// Severity Badge Component
const SeverityBadge = ({ severity }) => {
  const colors = {
    Severe: 'bg-red-500 text-white',
    Moderate: 'bg-amber-500 text-white',
    Minor: 'bg-green-500 text-white'
  };
  
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${colors[severity] || 'bg-gray-500 text-white'}`}>
      {severity}
    </span>
  );
};

// Damage Card Component
const DamageCard = ({ damage, index }) => {
  const borderColors = {
    Severe: 'border-l-red-500',
    Moderate: 'border-l-amber-500',
    Minor: 'border-l-green-500'
  };

  return (
    <div className={`damage-card bg-white rounded-lg shadow-md p-4 border-l-4 ${borderColors[damage.severity]}`}>
      <div className="flex justify-between items-start mb-2">
        <h4 className="font-semibold text-gray-800 capitalize">
          {damage.class_name.replace('_', ' ')}
        </h4>
        <SeverityBadge severity={damage.severity} />
      </div>
      <div className="space-y-2 text-sm text-gray-600">
        <div className="flex justify-between">
          <span>Confidence:</span>
          <span className="font-medium">{(damage.confidence * 100).toFixed(1)}%</span>
        </div>
        {damage.area_percentage && (
          <div className="flex justify-between">
            <span>Damage Area:</span>
            <span className="font-medium">{damage.area_percentage}%</span>
          </div>
        )}
        <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
          <div 
            className={`h-2 rounded-full ${
              damage.severity === 'Severe' ? 'bg-red-500' :
              damage.severity === 'Moderate' ? 'bg-amber-500' : 'bg-green-500'
            }`}
            style={{ width: `${damage.confidence * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
};

// Image Viewer Component
const ImageViewer = ({ title, imageSrc, icon: Icon }) => (
  <div className="bg-white rounded-xl shadow-lg overflow-hidden">
    <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-4 py-3 flex items-center gap-2">
      <Icon className="w-5 h-5 text-white" />
      <h3 className="text-white font-semibold">{title}</h3>
    </div>
    <div className="p-4">
      <img 
        src={imageSrc}
        alt={title}
        className="w-full h-auto rounded-lg shadow-inner"
      />
    </div>
  </div>
);

// Stats Component
const StatsBar = ({ result }) => {
  const stats = [
    { label: 'Damages Found', value: result.total_damages, icon: AlertTriangle, color: 'text-red-500' },
    { label: 'Processing Time', value: `${result.processing_time_ms}ms`, icon: Clock, color: 'text-blue-500' },
    { label: 'Image ID', value: result.image_id, icon: Camera, color: 'text-purple-500' },
  ];

  return (
    <div className="grid grid-cols-3 gap-4 mb-6">
      {stats.map((stat, idx) => (
        <div key={idx} className="bg-white rounded-lg shadow-md p-4 text-center">
          <stat.icon className={`w-6 h-6 mx-auto mb-2 ${stat.color}`} />
          <div className="text-2xl font-bold text-gray-800">{stat.value}</div>
          <div className="text-sm text-gray-500">{stat.label}</div>
        </div>
      ))}
    </div>
  );
};

// Main App Component
function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [viewMode, setViewMode] = useState('detection'); // 'detection' | 'segmentation'

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFile = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/detect`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze image. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 to-slate-200">
      {/* Header */}
      <header className="gradient-bg text-white py-6 shadow-lg">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-white/20 p-3 rounded-xl">
                <Car className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Car Damage Detection AI</h1>
                <p className="text-blue-100 text-sm">Powered by YOLOv8 Deep Learning</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="bg-green-400 w-2 h-2 rounded-full animate-pulse"></span>
              <span>Model Active</span>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Upload Section */}
        {!result && (
          <div className="max-w-2xl mx-auto mb-8">
            <div
              className={`upload-zone rounded-2xl p-12 text-center bg-white shadow-lg
                ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {preview ? (
                <div className="space-y-4">
                  <img 
                    src={preview} 
                    alt="Preview" 
                    className="max-h-64 mx-auto rounded-lg shadow-md"
                  />
                  <p className="text-gray-600">{file?.name}</p>
                  <div className="flex justify-center gap-4">
                    <button
                      onClick={analyzeImage}
                      disabled={loading}
                      className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-3 
                        rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all
                        disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Eye className="w-5 h-5" />
                          Analyze Damage
                        </>
                      )}
                    </button>
                    <button
                      onClick={resetAll}
                      className="bg-gray-200 text-gray-700 px-6 py-3 rounded-xl font-semibold
                        hover:bg-gray-300 transition-all flex items-center gap-2"
                    >
                      <X className="w-5 h-5" />
                      Clear
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <Upload className="w-16 h-16 mx-auto text-blue-500 mb-4" />
                  <h3 className="text-xl font-semibold text-gray-700 mb-2">
                    Drop your car image here
                  </h3>
                  <p className="text-gray-500 mb-4">or click to browse files</p>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="inline-block bg-blue-600 text-white px-6 py-3 rounded-xl 
                      font-semibold cursor-pointer hover:bg-blue-700 transition-colors"
                  >
                    Select Image
                  </label>
                </>
              )}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="max-w-2xl mx-auto mb-8 bg-red-50 border border-red-200 rounded-xl p-4 flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-red-500" />
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Stats */}
            <StatsBar result={result} />

            {/* View Mode Toggle */}
            <div className="flex justify-center gap-2 mb-6">
              <button
                onClick={() => setViewMode('detection')}
                className={`px-6 py-2 rounded-xl font-medium transition-all flex items-center gap-2
                  ${viewMode === 'detection' 
                    ? 'bg-blue-600 text-white shadow-lg' 
                    : 'bg-white text-gray-600 hover:bg-gray-100'}`}
              >
                <Eye className="w-4 h-4" />
                Detection View
              </button>
              <button
                onClick={() => setViewMode('segmentation')}
                className={`px-6 py-2 rounded-xl font-medium transition-all flex items-center gap-2
                  ${viewMode === 'segmentation' 
                    ? 'bg-blue-600 text-white shadow-lg' 
                    : 'bg-white text-gray-600 hover:bg-gray-100'}`}
                disabled={!result.segmentation_mask}
              >
                <Layers className="w-4 h-4" />
                Segmentation View
              </button>
              <button
                onClick={resetAll}
                className="px-6 py-2 rounded-xl font-medium bg-gray-200 text-gray-700 
                  hover:bg-gray-300 transition-all flex items-center gap-2"
              >
                <Camera className="w-4 h-4" />
                New Analysis
              </button>
            </div>

            {/* Images Grid */}
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <ImageViewer 
                title="Original Image" 
                imageSrc={`data:image/jpeg;base64,${result.original_image}`}
                icon={Camera}
              />
              <ImageViewer 
                title={viewMode === 'detection' ? 'Detection Results' : 'Segmentation Mask'}
                imageSrc={`data:image/jpeg;base64,${
                  viewMode === 'detection' ? result.annotated_image : result.segmentation_mask
                }`}
                icon={viewMode === 'detection' ? Eye : Layers}
              />
            </div>

            {/* Damages List */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="w-6 h-6 text-blue-600" />
                <h3 className="text-xl font-bold text-gray-800">Damage Analysis Report</h3>
              </div>
              
              {result.damages.length > 0 ? (
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {result.damages.map((damage, idx) => (
                    <DamageCard key={idx} damage={damage} index={idx} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <CheckCircle className="w-16 h-16 mx-auto text-green-500 mb-4" />
                  <h4 className="text-xl font-semibold text-gray-700">No Damage Detected</h4>
                  <p className="text-gray-500">The vehicle appears to be in good condition.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p>Car Damage Detection AI • Built with YOLOv8, FastAPI & React</p>
          <p className="text-sm mt-1">Deep Learning Project © 2025</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
