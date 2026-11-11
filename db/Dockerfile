# PostgreSQL 16 with Apache AGE (the graph) and pgvector (the docs index): one database, two models.
FROM apache/age:release_PG16_1.6.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-16-pgvector \
 && rm -rf /var/lib/apt/lists/*
