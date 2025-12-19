import process from 'node:process';
import path from 'node:path';
import { createClient } from '@supabase/supabase-js';

function parseArgs(argv) {
  const out = { _: [] };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) {
      out._.push(a);
      continue;
    }
    const key = a.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

function usage(msg) {
  if (msg) console.error(msg);
  console.error(
    [
      'Usage:',
      '  node scripts/seed-supabase.mjs --project-name <name> --subject-dir <path> --email <email> --password <password>',
      '',
      'Env (choose one mode):',
      '  Mode A (preferred, admin):',
      '    SUPABASE_URL',
      '    SUPABASE_SERVICE_ROLE_KEY   (Supabase dashboard: API keys → Secret key / service_role equivalent)',
      '',
      '  Mode B (user, no admin):',
      '    SUPABASE_URL',
      '    SUPABASE_ANON_KEY',
      '    (script will sign up/sign in with --email/--password and seed rows as that user)',
      '',
      'Notes:',
      '  - Never put any secret key into VITE_* or GitHub Pages env.',
      '  - This script seeds DB rows in tables: projects, subjects.',
    ].join('\n')
  );
  process.exit(1);
}

function requireArg(args, name) {
  const v = args[name];
  if (!v || typeof v !== 'string') usage(`Missing --${name}`);
  return v;
}

async function getOrCreateUserAdmin(sbAdmin, email, password) {
  // Try to find existing user (best-effort)
  try {
    const { data } = await sbAdmin.auth.admin.listUsers({ perPage: 200, page: 1 });
    const existing = (data?.users || []).find(u => (u.email || '').toLowerCase() === email.toLowerCase());
    if (existing) return existing;
  } catch {
    // ignore (API shape may vary)
  }

  const { data, error } = await sbAdmin.auth.admin.createUser({
    email,
    password,
    email_confirm: true,
  });
  if (error) throw error;
  if (!data?.user) throw new Error('Failed to create user');
  return data.user;
}

async function ensureSignedInUser(sbUser, email, password) {
  // Try sign in first
  {
    const { data, error } = await sbUser.auth.signInWithPassword({ email, password });
    if (!error && data?.session) return data.session;
  }

  // Otherwise sign up (may require email confirmations depending on project settings)
  {
    const { data, error } = await sbUser.auth.signUp({ email, password });
    if (error) throw error;
    if (data?.session) return data.session;
  }

  throw new Error(
    'Sign-up succeeded but no session was returned (email confirmation may be required). Disable confirmations or use admin mode.'
  );
}

async function main() {
  const args = parseArgs(process.argv);

  const projectName = requireArg(args, 'project-name');
  const subjectDir = requireArg(args, 'subject-dir');
  const email = requireArg(args, 'email');
  const password = requireArg(args, 'password');

  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const anonKey = process.env.SUPABASE_ANON_KEY;

  if (!supabaseUrl) usage('Missing SUPABASE_URL');

  const subjectName = path.basename(subjectDir);

  if (serviceRoleKey) {
    // Admin mode: create user and seed rows for that user.
    const sbAdmin = createClient(supabaseUrl, serviceRoleKey, {
      auth: { persistSession: false, autoRefreshToken: false },
    });

    const user = await getOrCreateUserAdmin(sbAdmin, email, password);
    const userId = user.id;

    const { data: projectRow, error: pErr } = await sbAdmin
      .from('projects')
      .insert({ user_id: userId, name: projectName, storage_path: '', copy_data_into_project: false })
      .select('*')
      .single();
    if (pErr) throw pErr;

    const { data: subjectRow, error: sErr } = await sbAdmin
      .from('subjects')
      .insert({
        user_id: userId,
        project_id: projectRow.id,
        name: subjectName,
        source_path: subjectDir,
        has_t1: false,
        has_dce: false,
        has_diffusion: false,
        stage_statuses: {},
      })
      .select('*')
      .single();
    if (sErr) throw sErr;

    console.log('Seeded (admin mode)');
    console.log(`User: ${user.email} (${userId})`);
    console.log(`Project: ${projectRow.name} (${projectRow.id})`);
    console.log(`Subject: ${subjectRow.name} (${subjectRow.id})`);
    return;
  }

  if (!anonKey) usage('Missing SUPABASE_SERVICE_ROLE_KEY and SUPABASE_ANON_KEY');

  // User mode: sign in/sign up, then seed rows under RLS.
  const sbUser = createClient(supabaseUrl, anonKey);
  await ensureSignedInUser(sbUser, email, password);

  const { data: projectRow, error: pErr } = await sbUser
    .from('projects')
    .insert({ name: projectName, storage_path: '', copy_data_into_project: false })
    .select('*')
    .single();
  if (pErr) throw pErr;

  const { data: subjectRow, error: sErr } = await sbUser
    .from('subjects')
    .insert({
      project_id: projectRow.id,
      name: subjectName,
      source_path: subjectDir,
      has_t1: false,
      has_dce: false,
      has_diffusion: false,
      stage_statuses: {},
    })
    .select('*')
    .single();
  if (sErr) throw sErr;

  console.log('Seeded (user mode)');
  console.log(`Project: ${projectRow.name} (${projectRow.id})`);
  console.log(`Subject: ${subjectRow.name} (${subjectRow.id})`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
