-- Reverse of contract.sql, for a rollback to an image from before the contract step only: the two legacy views again, with
-- their frozen names and types, as projections of the live generation's typed columns, owned by rag_indexer so the old
-- indexer's publish can repoint them; rag_app's default privileges and CREATE on public back.
-- The older assistant images read only paragraph_iid from the child metadata. The older indexer cannot roll back to or compare
-- a generation without the legacy columns (every generation at or after the cut): it refuses with UndefinedColumn
-- langchain_id, views unchanged; roll the index back with the indexer of the contract step before this script, or let the
-- old indexer build. It casts chunk_id to uuid; the stable chunk ids are 32 hex characters, which Postgres accepts as uuid
-- input. It refuses unless the cut has run on the live generation. Run as the bootstrap superuser, in one transaction:
--   psql -U vector_user -d vector_db -X -v ON_ERROR_STOP=1 --single-transaction -f contract_reverse.sql
SET lock_timeout = '10s';
DO $$
DECLARE
    t jsonb;
BEGIN
    SELECT tables INTO t FROM index_generations WHERE collection = 'rag' AND status = 'live' AND layout = 'v2';
    IF t IS NULL OR (SELECT count(*) FROM index_generations WHERE collection = 'rag' AND status = 'live') <> 1 THEN
        RAISE EXCEPTION 'reverse refused: want exactly one live generation of rag, with layout v2';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid = format('public.%I', t ->> 'children')::regclass
               AND attname = 'langchain_id' AND NOT attisdropped) THEN
        RAISE EXCEPTION 'reverse refused: % still has langchain_id (the cut has not run on it); the legacy views need no reverse',
            t ->> 'children';
    END IF;
    EXECUTE format('CREATE OR REPLACE VIEW public.rag_production AS SELECT chunk_id::uuid AS langchain_id, content, '
                   'embedding, json_build_object(''paragraph_iid'', section_id) AS langchain_metadata, fts AS content_tsv '
                   'FROM public.%I', t ->> 'children');
    EXECUTE format('CREATE OR REPLACE VIEW public.rag_production_parents AS SELECT section_id AS paragraph_iid, '
                   'content, metadata FROM public.%I', t ->> 'parents');
END
$$;
ALTER VIEW public.rag_production OWNER TO rag_indexer;
ALTER VIEW public.rag_production_parents OWNER TO rag_indexer;
GRANT SELECT ON public.rag_production, public.rag_production_parents TO rag_app;
ALTER DEFAULT PRIVILEGES FOR ROLE rag_indexer IN SCHEMA public GRANT SELECT ON TABLES TO rag_app;
GRANT CREATE ON SCHEMA public TO rag_app;
