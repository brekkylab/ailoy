//! Opening the database, asking it, and turning the answer into a table. Shared by all
//! three commands.

use std::path::Path;

use rusqlite::{Connection, OpenFlags, limits::Limit, types::ValueRef};

use super::Failure;

/// One cell, as one of SQLite's five storage classes.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Value {
    /// How it is written in a table.
    ///
    /// `NULL` is an empty cell because that is what "no value" reads as in a table.
    /// Writing `"NULL"` would not be distinguishable from text holding those four letters.
    pub fn cell(&self) -> String {
        match self {
            Value::Null => String::new(),
            Value::Integer(n) => n.to_string(),
            Value::Real(f) => {
                // A whole-number float is written `6.0`, not `6`. The view's
                // `real_years` is a float, and `6` would read as a month count.
                if f.fract() == 0.0 {
                    format!("{f:.1}")
                } else {
                    f.to_string()
                }
            }
            Value::Text(s) => s.clone(),
            Value::Blob(b) => format!("<{} bytes>", b.len()),
        }
    }
}

/// What a query answered with.
pub struct Rows {
    pub columns: Vec<String>,
    /// At most `limit` of them.
    pub rows: Vec<Vec<Value>>,
    /// How many rows actually matched, which is not `rows.len()` when it was cut short.
    ///
    /// This is the whole reason the count is carried separately: an answer that was
    /// truncated and does not say so reads as the complete one, and the caller who should
    /// have asked a narrower question never learns it.
    pub total: usize,
}

impl Rows {
    pub fn truncated(&self) -> bool {
        self.total > self.rows.len()
    }
}

/// Opens the pool: read-only, and unable to attach another database.
///
/// # Why the SQL is not inspected
///
/// Opening the connection `SQLITE_OPEN_READ_ONLY` means a write is refused by SQLite
/// itself. Guessing which statements write is the part that goes wrong: `WITH … INSERT`,
/// a trigger behind a `SELECT`, a `PRAGMA` that mutates — every one of them is a
/// statement some parser called a read.
///
/// # `ATTACH` is not stopped by read-only
///
/// Attaching is not itself a write, so it goes through — and then another database
/// outside the mount becomes readable. This example claims the pool is reached through
/// this command alone, so it is closed.
///
/// # `journal_mode` is left alone
///
/// The file may sit on a FUSE mount, and the WAL index is a shared-memory file every
/// connection expects to be coherent between processes — which FUSE is under no
/// obligation to provide. The failure is silent. cortex's `sqlite` does the same, for the
/// same reason.
pub fn open(path: &Path) -> Result<Connection, Failure> {
    let conn =
        Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| {
            // A missing file is caught here: read-only does not create one.
            Failure::failed(format!("headhunting: cannot open the pool: {e}"))
        })?;
    conn.set_limit(Limit::SQLITE_LIMIT_ATTACHED, 0)
        .map_err(|e| Failure::failed(format!("headhunting: {e}")))?;
    Ok(conn)
}

/// Runs `sql` and returns at most `limit` rows.
pub fn ask(conn: &Connection, sql: &str, limit: usize) -> Result<Rows, rusqlite::Error> {
    ask_with(conn, sql, &[], limit)
}

/// Runs it with values bound.
///
/// Injection is not the only reason values are not concatenated into the SQL. Candidate
/// names and cities carry quotes, hyphens, and Hangul, so whoever concatenates has to
/// vouch every time that all of it was escaped correctly.
pub fn ask_with(
    conn: &Connection,
    sql: &str,
    binds: &[String],
    limit: usize,
) -> Result<Rows, rusqlite::Error> {
    let mut stmt = conn.prepare(sql)?;
    let columns: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();
    let width = columns.len();

    let mut rows = Vec::new();
    let mut total = 0usize;
    let mut cursor = stmt.query(rusqlite::params_from_iter(binds))?;
    // Every row is stepped through even past the limit, because `total` is the count of
    // what matched and SQLite will not say what it has not been asked for. Only rows under
    // the limit are kept, so the cost of the rest is the step and not the allocation.
    while let Some(row) = cursor.next()? {
        total += 1;
        if rows.len() < limit {
            let mut cells = Vec::with_capacity(width);
            for i in 0..width {
                cells.push(match row.get_ref(i)? {
                    ValueRef::Null => Value::Null,
                    ValueRef::Integer(n) => Value::Integer(n),
                    ValueRef::Real(f) => Value::Real(f),
                    ValueRef::Text(b) => Value::Text(String::from_utf8_lossy(b).into_owned()),
                    ValueRef::Blob(b) => Value::Blob(b.to_vec()),
                });
            }
            rows.push(cells);
        }
    }
    Ok(Rows {
        columns,
        rows,
        total,
    })
}

/// How many columns this text occupies on screen.
///
/// Hangul, Han, and kana are two columns wide. Aligning by `char` count instead skews
/// any column holding a Korean name — and in this pool a large share of the 600 have one,
/// so that is the common case, not the rare one.
fn width(s: &str) -> usize {
    s.chars()
        .map(|c| match c as u32 {
            0x1100..=0x115F        // Hangul jamo
            | 0x2E80..=0x303E      // CJK radicals and symbols
            | 0x3041..=0x33FF      // kana, Hangul compatibility jamo, CJK compatibility
            | 0x3400..=0x4DBF      // CJK extension A
            | 0x4E00..=0x9FFF      // CJK unified ideographs
            | 0xA000..=0xA4CF      // Yi syllables
            | 0xAC00..=0xD7A3      // Hangul syllables
            | 0xF900..=0xFAFF      // CJK compatibility ideographs
            | 0xFE30..=0xFE6F      // CJK compatibility forms
            | 0xFF00..=0xFF60      // fullwidth forms
            | 0xFFE0..=0xFFE6 => 2,
            _ => 1,
        })
        .sum()
}

/// A table: one header line, the rows, and — only when truncated — how many there were.
///
/// It is aligned because both a person and an agent read it. Tabs are readable by the
/// machine but the eye cannot follow a column, and this table also flows onto the screen
/// a person watches the run on.
pub fn table(rows: &Rows) -> String {
    if rows.columns.is_empty() {
        return String::new();
    }

    let cells: Vec<Vec<String>> = rows
        .rows
        .iter()
        .map(|r| r.iter().map(Value::cell).collect())
        .collect();

    let widths: Vec<usize> = rows
        .columns
        .iter()
        .enumerate()
        .map(|(i, name)| {
            cells
                .iter()
                .map(|r| width(&r[i]))
                .chain(std::iter::once(width(name)))
                .max()
                .unwrap_or(0)
        })
        .collect();

    let mut out = String::new();
    let mut line = |values: &[String]| {
        let last = values.len() - 1;
        for (i, v) in values.iter().enumerate() {
            out.push_str(v);
            // The last column is not padded. Padding would leave trailing spaces, and
            // stripping them would become the job of whoever receives this as a file.
            if i != last {
                out.push_str(&" ".repeat(widths[i].saturating_sub(width(v)) + 2));
            }
        }
        out.push('\n');
    };

    line(&rows.columns);
    for row in &cells {
        line(row);
    }
    if rows.truncated() {
        out.push_str(&format!("-- {} of {} rows\n", rows.rows.len(), rows.total));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executable::tests::pool;

    #[test]
    fn a_korean_name_does_not_skew_the_columns() {
        let rows = Rows {
            columns: vec!["name".into(), "city".into()],
            rows: vec![
                vec![Value::Text("지훈 도".into()), Value::Text("Seoul".into())],
                vec![Value::Text("Blake".into()), Value::Text("Berlin".into())],
            ],
            total: 2,
        };
        let out = table(&rows);
        // `지훈 도` is 4 chars but 7 columns wide. The column the second value starts
        // at must be the same on both lines.
        let starts: Vec<usize> = out
            .lines()
            .skip(1)
            .zip(["Seoul", "Berlin"])
            .map(|(line, city)| width(&line[..line.find(city).expect("the city is there")]))
            .collect();
        assert_eq!(starts[0], starts[1], "{out}");
    }

    #[test]
    fn null_is_an_empty_cell_not_the_word() {
        assert_eq!(Value::Null.cell(), "");
    }

    /// The view's `real_years` must still read as years when it lands on a whole number.
    #[test]
    fn a_whole_number_of_years_still_reads_as_years() {
        assert_eq!(Value::Real(6.0).cell(), "6.0");
    }

    #[test]
    fn opening_refuses_to_create_a_database_that_is_not_there() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.db");
        assert!(open(&path).is_err());
        assert!(!path.exists(), "read-only must not create the file");
    }

    #[test]
    fn the_connection_refuses_to_attach() {
        let (_dir, db) = pool();
        let conn = open(&db).unwrap();
        let sql = format!("ATTACH DATABASE '{}' AS other", db.display());
        assert!(ask(&conn, &sql, 10).is_err());
    }
}
