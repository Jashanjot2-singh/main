# Asset Reminder API

A Django REST API for managing assets with two critical timelines:
- **Service Time**: When an asset is due for maintenance.
- **Expiration Time**: When an asset becomes invalid or unusable.

The system automatically sends **desktop notifications** 15 minutes before these deadlines using `plyer`, and logs:
- **Reminders** in a `Notification` table
- **Missed services/expirations** in a `Violation` table

> Includes automatic background checking, RESTful API endpoints, Swagger/Postman documentation, and optional MySQL support.

---

## 🚀 Features

- Add/update/delete assets with `service_time` and `expiration_time`
- Live desktop reminders 15 minutes before deadlines ⏰
- Logs reminders in `Notification` table
- Logs missed deadlines in `Violation` table
- Periodic check runs automatically in the background
- REST API endpoints with Swagger UI and Postman support

---

## 📦 Tech Stack

- **Backend**: Python, Django, Django REST Framework
- **Notifications**: Plyer (for system notifications)
- **Database**: SQLite (default), supports MySQL (XAMPP)
- **Documentation**: Postman Collection / Swagger (drf-yasg optional)

---

## 🛠 Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/yourname/asset-reminder-api.git
cd asset-reminder-api
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Migrate Database
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Run the Server
```bash
python manage.py runserver
```

### 6. Use the API (Postman)
Import the provided `Asset_Reminder_API.postman_collection.json` into Postman.

---

## 🔁 Background Reminder Thread
The reminder engine runs every **60 seconds** in a background thread.

```python
# apps.py
threading.Thread(target=run_checks_loop, daemon=True).start()
```

This thread:
- Checks all assets
- If within 15 minutes of `service_time` or `expiration_time`, triggers a notification and logs to `Notification`
- If the asset is already overdue and not serviced, logs to `Violation`

---

## 🔔 Example Notification

> If an asset has `service_time` in the next 15 minutes, the system will display:

```
🔔 Service Reminder: Pump A
Service due at 2025-06-26 22:00
```

---

## 🔗 API Endpoints

| Method | Endpoint             | Description                       |
|--------|----------------------|-----------------------------------|
| POST   | `/assets/`           | Create a new asset                |
| GET    | `/assets/`           | List all assets                   |
| GET    | `/assets/<id>/`      | Get asset by ID                   |
| PATCH  | `/assets/<id>/`      | Update asset (partial)            |
| DELETE | `/assets/<id>/`      | Delete asset                      |
| POST   | `/run-checks/`       | Trigger reminder + violation logic|
| GET    | `/notifications/`    | View all reminders logged         |
| GET    | `/violations/`       | View all violations logged        |

---

## ✅ Validation Logic
- `service_time` must be **before** `expiration_time`
- `is_serviced = False` means the asset is still due
- Duplicate notifications or violations are avoided using `.get_or_create()`

---

## 📁 Database Models

### Asset
```python
name: str
service_time: datetime
expiration_time: datetime
is_serviced: bool
```

### Notification
```python
asset: FK(Asset)
type: 'service' | 'expiration'
timestamp: auto_now_add
```

### Violation
```python
asset: FK(Asset)
type: 'service' | 'expiration'
timestamp: auto_now_add
```

---

## 🧪 Unit Tests
Run all unit tests with:
```bash
python manage.py test
```
Tests include:
- Asset creation & validation
- Notification generation logic
- Violation logging

---

## 📄 License
This project is for educational and professional demo purposes.

---

## 👨‍💻 Author
**Jashan Jot**  

---

