output "airflow_url" {
  value = "http://${aws_instance.airflow.public_ip}:8080"
}

output "dashboard_url" {
  value = "http://${aws_instance.airflow.public_ip}:8501"
}

output "readouts_bucket" {
  value = aws_s3_bucket.readouts.bucket
}

output "instance_id" {
  value = aws_instance.airflow.id
}

output "admin_password_command" {
  description = "Prints the Airflow admin password (user: admin)."
  value       = "aws ssm get-parameter --region ${var.region} --with-decryption --name ${aws_ssm_parameter.airflow_admin.name} --query Parameter.Value --output text"
}

output "shell_command" {
  description = "Open a shell on the instance through Session Manager, no SSH needed."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.airflow.id}"
}
