-- Run once in the Supabase SQL editor (or `supabase db execute -f schema.sql`)
-- before running ingest.py.

create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists chunks (
    id uuid primary key default gen_random_uuid(),
    chapter_id text not null,
    chunk_type text not null check (chunk_type in ('vocabulary', 'grammar', 'reading')),
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(512),  -- voyage-3-lite dimension; change if you swap embedding models
    created_at timestamptz not null default now(),
    unique (chapter_id, chunk_type, content)
);

create index if not exists chunks_chapter_type_idx on chunks (chapter_id, chunk_type);

create index if not exists chunks_embedding_idx
    on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
