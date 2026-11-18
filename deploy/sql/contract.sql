-- The contract step of the index. The two legacy views and the legacy columns of the live and the previous generation go;
-- rag_app reads the index through grants on the two typed views and index_generations; tables created after the cut give
-- it no privilege (no default privilege, no CREATE on public); its grants on tables that existed at the cut stay until they
-- are dropped. Run once, as the bootstrap superuser, in one transaction, after the images that read the typed views are
-- live and no older image runs:
--   psql -U vector_user -d vector_db -X -v ON_ERROR_STOP=1 --single-transaction -f contract.sql
-- A second run changes nothing. Undo: contract_reverse.sql. The collection is rag.
SET lock_timeout = '10s';
DO $$
DECLARE
    g record;
    live int := 0;
    previous int := 0;
    bad bigint;
BEGIN
    FOR g IN SELECT generation, status, layout FROM index_generations
             WHERE collection = 'rag' AND status IN ('live', 'previous') LOOP
        IF g.layout <> 'v2' THEN
            RAISE EXCEPTION 'contract refused: generation % (%) has layout %; publish a v2 generation first',
                g.generation, g.status, g.layout;
        END IF;
        IF g.status = 'live' THEN
            live := live + 1;
        ELSE
            previous := previous + 1;
        END IF;
    END LOOP;
    IF live <> 1 OR previous < 1 THEN
        RAISE EXCEPTION 'contract refused: % live and % previous generations of rag (want 1 and at least 1)',
            live, previous;
    END IF;
    DROP VIEW IF EXISTS public.rag_production;
    DROP VIEW IF EXISTS public.rag_production_parents;
    FOR g IN SELECT tables ->> 'children' AS chunks FROM index_generations
             WHERE collection = 'rag' AND status IN ('live', 'previous') LOOP
        IF EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid = format('public.%I', g.chunks)::regclass
                   AND attname = 'langchain_metadata' AND NOT attisdropped) THEN
            EXECUTE format('SELECT count(*) FROM public.%I WHERE section_id IS DISTINCT FROM langchain_metadata ->> ''paragraph_iid'' '
                           'OR chunk_id IS DISTINCT FROM langchain_id::text', g.chunks) INTO bad;
            IF bad > 0 THEN
                RAISE EXCEPTION 'contract refused: % rows of % differ from their legacy columns (section_id, chunk_id)', bad, g.chunks;
            END IF;
        END IF;
        EXECUTE format('ALTER TABLE public.%I DROP COLUMN IF EXISTS langchain_id, DROP COLUMN IF EXISTS langchain_metadata',
                       g.chunks);
    END LOOP;
END
$$;
GRANT SELECT ON public.rag_chunks, public.rag_sections, public.index_generations TO rag_app;
ALTER DEFAULT PRIVILEGES FOR ROLE rag_indexer IN SCHEMA public REVOKE SELECT ON TABLES FROM rag_app;
REVOKE CREATE ON SCHEMA public FROM rag_app;
