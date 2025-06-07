import React from 'react';
import { SegmentData } from '../../models/classificator-responses';

interface TooltipProps {
  segment: SegmentData;
  index: number;
}

const formatNum = (num: number | null): string => {
    return num?.toFixed(0) ?? 'N/A';
}

const SegmentTooltip: React.FC<TooltipProps> = ({ segment, index }) => {
    if (segment.is_good === null && segment.branching_point == null) {
        return null;
    }

    const isGoodText = segment.is_good == null ? "Unknown Value" :  (segment.is_good ? "GOOD" : "BAD")

   return (
      <div className="absolute bottom-full left-1/2 mb-1 px-3 py-1.5
                      -translate-x-1/2 translate-y-2
                      bg-gray-900/95 text-white text-xs font-mono
                      rounded-md shadow-lg
                      opacity-0 invisible
                      group-hover:opacity-100 group-hover:visible group-hover:translate-y-0
                      transition-all duration-75 ease-in-out
                      z-20 whitespace-nowrap pointer-events-none
                      lg:w-44
                      ">
          <div className="font-bold text-sm mb-1 border-b border-gray-600 pb-1 text-wrap">Segment {index} ({isGoodText})</div>
          <p className="text-wrap">Branching Points: {formatNum(segment.branching_point)}</p>
         {/* Arrow indicator */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0
                border-l-[6px] border-l-transparent
                border-t-[6px] border-t-gray-900/95
                border-r-[6px] border-r-transparent">
           </div>
      </div>
   );
}
export default SegmentTooltip;