import React, { useState, useEffect, useMemo } from 'react';
import { ClassificationResult, SegmentData } from '../../models/classificator-responses';
import SegmentTooltip from './segment-tooltip';


interface GridCellProps {
  segment: SegmentData;
  index: number;
}


const GridCell: React.FC<GridCellProps> = React.memo(({ segment, index }) => {
  const baseColor = segment.is_good === true ? 'green'
    : segment.is_good === false ? 'red'
      : 'gray';
  const bgColorClass = baseColor === 'green' ? 'bg-green-500/20 hover:bg-green-500/40'
    : baseColor === 'red' ? 'bg-red-500/20 hover:bg-red-500/40'
      : 'bg-gray-500/10 hover:bg-gray-500/30';
  const borderColorClass = baseColor === 'green' ? 'border-green-700'
    : baseColor === 'red' ? 'border-red-700'
      : 'border-gray-500';

  return (
    <div className={`relative group border-[1px] transition-colors duration-200 ${borderColorClass} ${bgColorClass}`}>
      <SegmentTooltip segment={segment} index={index} />
    </div>
  );
});


interface ImageDisplayProps {
  imageUrl: string;
  analysisResult: ClassificationResult | null;
}


const ImageDisplay: React.FC<ImageDisplayProps> = ({ imageUrl, analysisResult }) => {
  const [naturalImageWidth, setNaturalImageWidth] = useState<number | null>(null);
  const [viewportWidth, setViewportWidth] = useState(window.innerWidth);

  // Listen for window resize to update viewport width for responsiveness
  useEffect(() => {
    const handleResize = () => setViewportWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Reset the image's natural width state when the image URL changes
  useEffect(() => {
    setNaturalImageWidth(null);
  }, [imageUrl]);

  // Once the image is loaded in the DOM, capture its natural width
  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    setNaturalImageWidth(e.currentTarget.naturalWidth);
  };

  // Calculate the container style based on image and viewport dimensions
  const containerStyle = useMemo((): React.CSSProperties => {
    // Before the image has loaded, use a default 90% screen width.
    if (!naturalImageWidth) {
      return { maxWidth: '95vw', margin: '0 auto' };
    }
    const screenWidth95 = viewportWidth * 0.9;
    const finalWidth = Math.min(naturalImageWidth, screenWidth95);

    return {
      width: `${finalWidth}px`,
      margin: '0 auto',
    };
  }, [naturalImageWidth, viewportWidth]);


  const gridStyle = useMemo((): React.CSSProperties => {
    if (!analysisResult) return {};
    return {
      display: 'grid',
      gridTemplateColumns: `repeat(${analysisResult.num_cols}, 1fr)`,
      gridTemplateRows: `repeat(${analysisResult.num_rows}, 1fr)`,
    };
  }, [analysisResult]);

  return (
    <div style={containerStyle} className="bg-white p-4 rounded-xl shadow-xl mt-6">
      <h3 className="text-xl font-semibold text-gray-800 mb-3 text-center">
        {analysisResult ? "Classification Result" : "Image Preview"}
        {analysisResult && (
          <span className="text-sm text-gray-500 ml-2">
            ({analysisResult.num_cols}x{analysisResult.num_rows} grid)
          </span>
        )}
      </h3>
      <div className="relative w-full border border-gray-200">
        <img
          src={imageUrl}
          alt="Selected"
          onLoad={handleImageLoad}
          className="block w-full h-auto"
        />

        {analysisResult && (
          <div
            style={gridStyle}
            className="absolute top-0 left-0 w-full h-full"
          >
            {analysisResult.segments.map((segment, index) => (
              <GridCell key={index} segment={segment} index={index} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageDisplay;