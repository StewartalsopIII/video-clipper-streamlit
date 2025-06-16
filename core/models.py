# core/models.py

from typing import List, Optional
from pydantic import BaseModel, Field

# --- Pydantic Models ---

class VideoAnalysis(BaseModel):
    summary: str = Field(..., description="A brief summary of the video content, in 2-3 sentences.")
    main_topics: List[str] = Field(
        ...,
        description="A list of distinct main topics or themes discussed in the video. Each topic should be a short phrase (2-5 words) suitable for guiding clip extraction.",
        min_length=1 # For Pydantic v2, actual enforcement might need validator or be on List itself
    )
    target_audience: Optional[str] = Field(None, description="The likely target audience for this video (e.g., 'Tech Enthusiasts', 'Business Leaders', 'Students').")

class Segment(BaseModel):
    start_time: int = Field(..., description="The start time of the segment in SECONDS from the beginning of the video.")
    end_time: int = Field(..., description="The end time of the segment in SECONDS from the beginning of the video.")
    title: str = Field(..., description="A concise, engaging title for this video segment (max 10 words).")
    description: str = Field(..., description="A brief 1-2 sentence description of what this segment is about.")

class SegmentResponse(BaseModel):
    segments: List[Segment] = Field(..., description="A list of identified video segments matching the criteria.")