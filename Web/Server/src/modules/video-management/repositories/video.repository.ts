export interface VideoRecord {
  id: string;
  matchId: string;
  filename: string;
  storagePath: string;
  contentType: string;
  sizeBytes: number;
  durationSeconds?: number;
  fps?: number;
  status: "pending_upload" | "uploaded" | "processing" | "ready" | "failed";
  createdAt: Date;
  updatedAt: Date;
}

export interface IVideoRepository {
  create(video: Omit<VideoRecord, "createdAt" | "updatedAt">): Promise<VideoRecord>;
  findById(id: string): Promise<VideoRecord | null>;
  findByMatchId(matchId: string): Promise<VideoRecord[]>;
  updateStatus(id: string, status: VideoRecord["status"]): Promise<VideoRecord | null>;
}

export class VideoRepository implements IVideoRepository {
  // In-memory backing store for modular monolith persistence; can be swapped with Postgres/Supabase table
  private videos = new Map<string, VideoRecord>();

  public async create(data: Omit<VideoRecord, "createdAt" | "updatedAt">): Promise<VideoRecord> {
    const record: VideoRecord = {
      ...data,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    this.videos.set(record.id, record);
    return record;
  }

  public async findById(id: string): Promise<VideoRecord | null> {
    return this.videos.get(id) || null;
  }

  public async findByMatchId(matchId: string): Promise<VideoRecord[]> {
    return Array.from(this.videos.values()).filter((v) => v.matchId === matchId);
  }

  public async updateStatus(id: string, status: VideoRecord["status"]): Promise<VideoRecord | null> {
    const record = this.videos.get(id);
    if (!record) return null;
    record.status = status;
    record.updatedAt = new Date();
    this.videos.set(id, record);
    return record;
  }
}
