## 🛠️ `systemctl` Commands Cheat Sheet

### 🔄 Service Management

```bash
# Start a service immediately
sudo systemctl start your-service

# Stop a running service
sudo systemctl stop your-service

# Restart a service
sudo systemctl restart your-service

# Reload config without restarting the service
sudo systemctl reload your-service
```

---

### 📌 Enable/Disable Service at Boot

```bash
# Enable service to start on boot
sudo systemctl enable your-service

# Disable service from starting on boot
sudo systemctl disable your-service

# Check if service is enabled
systemctl is-enabled your-service
```

---

### 📊 Status & Logs

```bash
# Check service status
systemctl status your-service

# Follow live logs from a service
journalctl -u your-service -f

# View logs with timestamps
journalctl -u your-service --since "1 hour ago"

# Show full logs of service
journalctl -u your-service
```

---

### 🧼 Daemon & Unit Management

```bash
# Reload systemd to recognize new/changed services
sudo systemctl daemon-reload

# List all loaded units (services, sockets, etc.)
systemctl list-units

# List all services (active and inactive)
systemctl list-units --type=service

# Mask a service (disable & prevent from being started manually)
sudo systemctl mask your-service

# Unmask a previously masked service
sudo systemctl unmask your-service
```

---

### 🧪 Debug & Testing

```bash
# Run service manually (for testing script behind)
sudo /usr/bin/your-service-script

# Check if systemd recognized your service unit
systemctl cat your-service

# Analyze boot performance
systemd-analyze blame
```

---

### 🔥 Clean Up

```bash
# Remove failed service units from list
systemctl reset-failed

# Show only failed services
systemctl --failed
```

---

### 📁 Service Unit File Locations

```bash
# System-wide services
/lib/systemd/system/
# User-defined or custom services
/etc/systemd/system/
```

---

> 📎 **Pro tip:** After editing `.service` files, always run:

```bash
sudo systemctl daemon-reload
```
