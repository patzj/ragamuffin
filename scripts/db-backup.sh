#!/usr/bin/env bash
set -euo pipefail

if docker compose version >/dev/null 2>&1; then
    compose=(docker compose)
else
    compose=(docker-compose)
fi

case "${1:-}:$#" in
    backup:2)
        umask 077
        archive=$2
        temporary=$(mktemp -- "$(dirname -- "$archive")/.ragamuffin-dump.XXXXXX")
        trap 'rm -f -- "$temporary"' EXIT
        "${compose[@]}" exec -T postgres sh -ec '
            exec pg_dump --format=custom --username="$POSTGRES_USER" --dbname="$POSTGRES_DB"
        ' > "$temporary"
        # A hard link publishes the completed dump without replacing an existing path.
        perl -e 'link($ARGV[0], $ARGV[1]) or die "Cannot create archive: $!\n"' "$temporary" "$archive"
        ;;
    restore:3)
        archive=$2
        target=$3
        [[ $target =~ ^[a-z][a-z0-9_]{0,62}$ && -f $archive && -r $archive ]] ||
            { echo "Expected a readable archive and a lowercase database name." >&2; exit 2; }
        "${compose[@]}" exec -T postgres sh -ec '
            if [ "$1" = "$POSTGRES_DB" ]; then
                echo "Cannot restore into the application database." >&2
                exit 2
            fi
            createdb --username="$POSTGRES_USER" --maintenance-db=postgres --template=template0 "$1"
            if ! pg_restore --username="$POSTGRES_USER" --dbname="$1" \
                --single-transaction --exit-on-error --no-owner --no-privileges; then
                echo "Restore failed; the new database may need manual cleanup." >&2
                exit 1
            fi
        ' restore "$target" < "$archive"
        ;;
    *)
        echo "Usage: $0 backup <archive> | restore <archive> <new-db>" >&2
        exit 2
        ;;
esac
