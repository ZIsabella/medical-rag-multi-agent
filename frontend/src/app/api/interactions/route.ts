import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

interface InteractionItem {
  id?: string | number;
  session_id?: string;
  query?: string;
  response?: string;
  created_at?: string;
  latency_ms?: number;
  is_safe?: boolean;
  safety_status?: string;
  status?: string;
  retrieval_items?: Array<{
    document_id?: string | number;
    title?: string;
    source?: string;
    similarity_score?: number;
    content?: string;
  }>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit') || '50';
    const offset = searchParams.get('offset') || '0';

    const dataServiceUrl = process.env.DATA_SERVICE_URL || 'http://data-service:8001';
    const targetUrl = `${dataServiceUrl}/api/v1/interactions?limit=${encodeURIComponent(limit)}&offset=${encodeURIComponent(offset)}`;

    const response = await fetch(targetUrl, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: `Data service responded with status ${response.status}: ${errorText}` },
        { status: response.status }
      );
    }

    const data: InteractionItem[] = await response.json();
    return NextResponse.json(data);
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return NextResponse.json(
      { error: `Failed to fetch interactions: ${errorMessage}` },
      { status: 500 }
    );
  }
}
