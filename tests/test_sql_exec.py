import sqlite3

import pytest

from lora_sql.sql_exec import execution_match, extract_sql, is_valid_on_ddl, normalize_sql, schema_from_sqlite


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);
        INSERT INTO products VALUES (1, 'pen', 1.5), (2, 'book', 12.0), (3, 'bag', 30.0);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_extract_sql_from_fence_and_trailing_text():
    assert extract_sql("```sql\nSELECT 1;\n```\nThis returns one.") == "SELECT 1"
    assert extract_sql("SELECT name FROM t; -- done") == "SELECT name FROM t"


def test_normalize_sql():
    assert normalize_sql('SELECT  name\nFROM t WHERE x = "a";') == "select name from t where x = 'a'"


def test_execution_match_ignores_row_order_without_order_by(db_path):
    gold = "SELECT name FROM products WHERE price > 1"
    pred = "SELECT name FROM products WHERE price > 1 ORDER BY name DESC"
    assert execution_match(db_path, gold, pred)["match"] is True


def test_execution_match_respects_order_by(db_path):
    gold = "SELECT name FROM products ORDER BY price"
    pred = "SELECT name FROM products ORDER BY price DESC"
    assert execution_match(db_path, gold, pred)["match"] is False


def test_invalid_prediction(db_path):
    res = execution_match(db_path, "SELECT 1", "SELEC nope")
    assert res["valid"] is False and res["match"] is False


def test_database_is_read_only(db_path):
    res = execution_match(db_path, "SELECT 1", "DELETE FROM products")
    assert res["valid"] is False
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 3


def test_schema_and_ddl_validation(db_path):
    schema = schema_from_sqlite(db_path)
    assert "CREATE TABLE products" in schema
    assert is_valid_on_ddl(schema, "SELECT name FROM products")
    assert not is_valid_on_ddl(schema, "SELECT missing_col FROM products")
