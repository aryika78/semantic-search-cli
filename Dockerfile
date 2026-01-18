# 1️⃣ Use official lightweight Python image
FROM python:3.11-slim

# 2️⃣ Set working directory inside container
WORKDIR /app

# 3️⃣ Copy dependency file first (for caching)
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy project code
COPY . .

# 6️⃣ Set default command (CLI entry)
ENTRYPOINT ["python", "-m", "cli.main"]
