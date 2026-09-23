import asyncio
import asyncpg


async def check_content():
    conn = await asyncpg.connect("postgresql://medical_user:medical_password@localhost:5432/medical_rag")

    empty_questions = await conn.fetchval("""
        SELECT COUNT(*) FROM documents 
        WHERE question IS NULL 
           OR question = '' 
           OR TRIM(question) = ''
    """)

    valid_questions = await conn.fetchval("""
        SELECT COUNT(*) FROM documents 
        WHERE question IS NOT NULL 
           AND TRIM(question) <> ''
    """)

    print(f"Rows with empty/null questions: {empty_questions}")
    print(f"Rows with valid questions: {valid_questions}")

    if valid_questions > 0:
        print("\n--- Sample of a VALID question ---")
        row = await conn.fetchrow(
            "SELECT question FROM documents WHERE question IS NOT NULL AND TRIM(question) <> '' LIMIT 1")
        print(f"'{row['question']}'")

    if empty_questions > 0:
        print("\n--- Sample of an EMPTY question ---")
        row = await conn.fetchrow(
            "SELECT question FROM documents WHERE question IS NULL OR question = '' OR TRIM(question) = '' LIMIT 1")
        print(f"'{row['question']}'")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(check_content())
