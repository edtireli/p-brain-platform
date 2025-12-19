-- Supabase schema for p-brain-web (projects/subjects/jobs)
-- Apply in the Supabase SQL editor.

-- Projects
create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid(),
  name text not null,
  storage_path text not null default '',
  copy_data_into_project boolean not null default true,
  config jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists projects_user_id_idx on public.projects (user_id);

-- Subjects
create table if not exists public.subjects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  name text not null,
  source_path text not null default '',
  has_t1 boolean not null default false,
  has_dce boolean not null default false,
  has_diffusion boolean not null default false,
  stage_statuses jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists subjects_project_id_idx on public.subjects (project_id);
create index if not exists subjects_user_id_idx on public.subjects (user_id);

-- Jobs (queue)
create table if not exists public.jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  subject_id uuid not null references public.subjects(id) on delete cascade,
  stage_id text not null,
  status text not null default 'queued',
  progress double precision not null default 0,
  current_step text not null default '',
  start_time timestamptz,
  end_time timestamptz,
  estimated_time_remaining double precision,
  error text,
  log_path text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists jobs_project_id_idx on public.jobs (project_id);
create index if not exists jobs_subject_id_idx on public.jobs (subject_id);
create index if not exists jobs_user_id_idx on public.jobs (user_id);
create index if not exists jobs_status_idx on public.jobs (status);

-- updated_at helpers
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_projects_updated_at on public.projects;
create trigger set_projects_updated_at
before update on public.projects
for each row execute function public.set_updated_at();

drop trigger if exists set_subjects_updated_at on public.subjects;
create trigger set_subjects_updated_at
before update on public.subjects
for each row execute function public.set_updated_at();

drop trigger if exists set_jobs_updated_at on public.jobs;
create trigger set_jobs_updated_at
before update on public.jobs
for each row execute function public.set_updated_at();

-- Row Level Security
alter table public.projects enable row level security;
alter table public.subjects enable row level security;
alter table public.jobs enable row level security;

-- Projects policies
drop policy if exists "projects_select_own" on public.projects;
create policy "projects_select_own"
  on public.projects for select
  using (user_id = auth.uid());

drop policy if exists "projects_insert_own" on public.projects;
create policy "projects_insert_own"
  on public.projects for insert
  with check (user_id = auth.uid());

drop policy if exists "projects_update_own" on public.projects;
create policy "projects_update_own"
  on public.projects for update
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

drop policy if exists "projects_delete_own" on public.projects;
create policy "projects_delete_own"
  on public.projects for delete
  using (user_id = auth.uid());

-- Subjects policies
drop policy if exists "subjects_select_own" on public.subjects;
create policy "subjects_select_own"
  on public.subjects for select
  using (user_id = auth.uid());

drop policy if exists "subjects_insert_own" on public.subjects;
create policy "subjects_insert_own"
  on public.subjects for insert
  with check (user_id = auth.uid());

drop policy if exists "subjects_update_own" on public.subjects;
create policy "subjects_update_own"
  on public.subjects for update
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

drop policy if exists "subjects_delete_own" on public.subjects;
create policy "subjects_delete_own"
  on public.subjects for delete
  using (user_id = auth.uid());

-- Jobs policies
drop policy if exists "jobs_select_own" on public.jobs;
create policy "jobs_select_own"
  on public.jobs for select
  using (user_id = auth.uid());

drop policy if exists "jobs_insert_own" on public.jobs;
create policy "jobs_insert_own"
  on public.jobs for insert
  with check (user_id = auth.uid());

drop policy if exists "jobs_update_own" on public.jobs;
create policy "jobs_update_own"
  on public.jobs for update
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

drop policy if exists "jobs_delete_own" on public.jobs;
create policy "jobs_delete_own"
  on public.jobs for delete
  using (user_id = auth.uid());
