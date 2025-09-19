# 📧 SMTP Configuration Guide for MailSift Ultra

This guide will help you configure SMTP settings to enable automatic license email delivery when payments are confirmed.

## 🔧 Environment Variables Setup

Add these environment variables to your `.env` file or deployment environment:

```bash
# SMTP Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
EMAIL_FROM=noreply@mailsift.com

# Alternative SMTP Providers
# For Outlook/Hotmail:
# SMTP_HOST=smtp-mail.outlook.com
# SMTP_PORT=587

# For Yahoo:
# SMTP_HOST=smtp.mail.yahoo.com
# SMTP_PORT=587

# For Custom SMTP:
# SMTP_HOST=mail.yourdomain.com
# SMTP_PORT=587
```

## 📋 Step-by-Step Configuration

### 1. **Gmail Configuration (Recommended)**

#### For Gmail:
1. **Enable 2-Factor Authentication** on your Google account
2. **Generate App Password**:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
   - Use this password (not your regular password)

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-16-character-app-password
EMAIL_FROM=your-email@gmail.com
```

### 2. **Outlook/Hotmail Configuration**

```bash
SMTP_HOST=smtp-mail.outlook.com
SMTP_PORT=587
SMTP_USER=your-email@outlook.com
SMTP_PASS=your-password
EMAIL_FROM=your-email@outlook.com
```

### 3. **Yahoo Mail Configuration**

```bash
SMTP_HOST=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USER=your-email@yahoo.com
SMTP_PASS=your-app-password
EMAIL_FROM=your-email@yahoo.com
```

### 4. **Custom SMTP Server**

For your own domain or hosting provider:

```bash
SMTP_HOST=mail.yourdomain.com
SMTP_PORT=587
SMTP_USER=noreply@yourdomain.com
SMTP_PASS=your-smtp-password
EMAIL_FROM=noreply@yourdomain.com
```

## 🚀 Popular Email Service Providers

### **SendGrid (Recommended for Production)**
```bash
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASS=your-sendgrid-api-key
EMAIL_FROM=noreply@yourdomain.com
```

### **Mailgun**
```bash
SMTP_HOST=smtp.mailgun.org
SMTP_PORT=587
SMTP_USER=postmaster@yourdomain.mailgun.org
SMTP_PASS=your-mailgun-password
EMAIL_FROM=noreply@yourdomain.com
```

### **Amazon SES**
```bash
SMTP_HOST=email-smtp.us-east-1.amazonaws.com
SMTP_PORT=587
SMTP_USER=your-ses-smtp-username
SMTP_PASS=your-ses-smtp-password
EMAIL_FROM=noreply@yourdomain.com
```

## 🔧 Deployment Configuration

### **Render.com**
Add environment variables in your Render dashboard:
1. Go to your service dashboard
2. Click "Environment" tab
3. Add each SMTP variable

### **Heroku**
```bash
heroku config:set SMTP_HOST=smtp.gmail.com
heroku config:set SMTP_PORT=587
heroku config:set SMTP_USER=your-email@gmail.com
heroku config:set SMTP_PASS=your-app-password
heroku config:set EMAIL_FROM=noreply@mailsift.com
```

### **DigitalOcean App Platform**
Add environment variables in your app settings:
1. Go to your app dashboard
2. Settings → Environment Variables
3. Add each SMTP variable

### **Docker**
Create a `.env` file:
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
EMAIL_FROM=noreply@mailsift.com
```

## 🧪 Testing SMTP Configuration

### **Test Email Function**
Add this to your application to test SMTP:

```python
def test_smtp_configuration():
    """Test SMTP configuration"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        
        smtp_host = os.environ.get('SMTP_HOST')
        smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        smtp_user = os.environ.get('SMTP_USER')
        smtp_pass = os.environ.get('SMTP_PASS')
        from_email = os.environ.get('EMAIL_FROM')
        
        if not all([smtp_host, smtp_user, smtp_pass]):
            return {"status": "error", "message": "SMTP configuration incomplete"}
        
        # Create test email
        msg = MIMEText("SMTP test successful!")
        msg['Subject'] = "MailSift SMTP Test"
        msg['From'] = from_email
        msg['To'] = smtp_user  # Send to yourself
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        
        return {"status": "success", "message": "SMTP test successful!"}
        
    except Exception as e:
        return {"status": "error", "message": f"SMTP test failed: {str(e)}"}
```

### **Manual Test**
```bash
# Test with curl
curl -X POST https://your-domain.com/api/test-smtp \
  -H "Content-Type: application/json" \
  -d '{"test": true}'
```

## 🔒 Security Best Practices

### **1. Use App Passwords**
- Never use your main account password
- Generate app-specific passwords
- Rotate passwords regularly

### **2. Environment Variables**
- Never commit SMTP credentials to code
- Use environment variables or secrets management
- Encrypt sensitive data in production

### **3. Rate Limiting**
- Implement rate limiting for email sending
- Monitor for abuse or spam
- Set daily email limits

### **4. Email Validation**
- Validate email addresses before sending
- Implement bounce handling
- Monitor delivery rates

## 📊 Monitoring and Analytics

### **Email Delivery Tracking**
```python
def track_email_delivery(email, status, error=None):
    """Track email delivery status"""
    # Log to database or monitoring service
    logger.info(f"Email to {email}: {status}")
    if error:
        logger.error(f"Email error: {error}")
```

### **SMTP Health Check**
```python
def smtp_health_check():
    """Check SMTP service health"""
    try:
        # Test connection
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            return {"status": "healthy", "response_time": "< 1s"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## 🚨 Troubleshooting

### **Common Issues**

#### **1. Authentication Failed**
```
Error: SMTPAuthenticationError
Solution: Check username/password, enable 2FA, use app password
```

#### **2. Connection Refused**
```
Error: SMTPConnectError
Solution: Check host/port, firewall settings, SSL/TLS configuration
```

#### **3. Timeout**
```
Error: SMTPTimeoutError
Solution: Check network connection, increase timeout, try different port
```

#### **4. TLS/SSL Issues**
```
Error: SMTPException
Solution: Use STARTTLS, check certificate validity
```

### **Debug Mode**
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
smtplib.DEBUG = True
```

## 📈 Production Recommendations

### **1. Use Dedicated Email Service**
- SendGrid, Mailgun, or Amazon SES
- Better deliverability
- Built-in analytics
- Compliance features

### **2. Implement Queue System**
- Use Celery or RQ for email queues
- Handle failures gracefully
- Retry failed emails
- Monitor queue health

### **3. Email Templates**
- Use professional templates
- Include unsubscribe links
- Comply with CAN-SPAM Act
- Test across email clients

### **4. Monitoring**
- Track delivery rates
- Monitor bounce rates
- Set up alerts for failures
- Regular health checks

## 🔗 Additional Resources

- [Gmail SMTP Settings](https://support.google.com/mail/answer/7126229)
- [SendGrid Documentation](https://sendgrid.com/docs/for-developers/sending-email/smtp-integration/)
- [Mailgun SMTP Guide](https://documentation.mailgun.com/en/latest/user_manual.html#smtp)
- [Amazon SES SMTP](https://docs.aws.amazon.com/ses/latest/dg/smtp-credentials.html)

## 📞 Support

If you need help with SMTP configuration:
- 🤖 **AI Support**: Visit `/support` for instant help
- 📧 **Email Support**: support@mailsift.com
- 📚 **Documentation**: Visit `/api/docs` for more guides

---

**Your SMTP configuration is now ready for automatic license email delivery!** 🎉
