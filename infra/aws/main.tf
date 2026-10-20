terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  bucket = "${var.project}-readouts-${data.aws_caller_identity.current.account_id}"
}

# -- results bucket -------------------------------------------------------------

resource "aws_s3_bucket" "readouts" {
  bucket        = local.bucket
  force_destroy = var.force_destroy_bucket
}

resource "aws_s3_bucket_versioning" "readouts" {
  bucket = aws_s3_bucket.readouts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "readouts" {
  bucket = aws_s3_bucket.readouts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "readouts" {
  bucket                  = aws_s3_bucket.readouts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -- Airflow admin password, kept out of user data and state outputs -----------

resource "random_password" "airflow_admin" {
  length  = 24
  special = false
}

resource "aws_ssm_parameter" "airflow_admin" {
  name  = "/${var.project}/airflow-admin-password"
  type  = "SecureString"
  value = random_password.airflow_admin.result
}

# -- instance role: Session Manager, the bucket, and that one parameter ---------

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "airflow" {
  name_prefix        = "${var.project}-"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.airflow.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "airflow" {
  statement {
    actions   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.readouts.arn, "${aws_s3_bucket.readouts.arn}/*"]
  }
  statement {
    actions   = ["ssm:GetParameter"]
    resources = [aws_ssm_parameter.airflow_admin.arn]
  }
}

resource "aws_iam_role_policy" "airflow" {
  name   = "readouts-and-secret"
  role   = aws_iam_role.airflow.id
  policy = data.aws_iam_policy_document.airflow.json
}

resource "aws_iam_instance_profile" "airflow" {
  name_prefix = "${var.project}-"
  role        = aws_iam_role.airflow.name
}

# -- network ----------------------------------------------------------------------

resource "aws_security_group" "airflow" {
  name_prefix = "${var.project}-"
  description = "Airflow UI and dashboard for the experimentation platform"
  vpc_id      = var.vpc_id

  ingress {
    description = "Airflow web UI"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  ingress {
    description = "Posterior dashboard"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# -- the machine that runs docker compose -----------------------------------------

resource "aws_instance" "airflow" {
  ami                         = data.aws_ssm_parameter.al2023.value
  instance_type               = var.instance_type
  subnet_id                   = var.subnet_id
  vpc_security_group_ids      = [aws_security_group.airflow.id]
  iam_instance_profile        = aws_iam_instance_profile.airflow.name
  associate_public_ip_address = true

  metadata_options {
    http_tokens = "required"
  }

  root_block_device {
    volume_size = var.disk_gb
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    repo_url       = var.repo_url
    repo_ref       = var.repo_ref
    region         = var.region
    s3_uri         = "s3://${aws_s3_bucket.readouts.bucket}/readouts"
    password_param = aws_ssm_parameter.airflow_admin.name
  })
  user_data_replace_on_change = true

  tags = {
    Name = "${var.project}-airflow"
  }
}

# Restart the instance on the same host if AWS's own hardware check fails.
resource "aws_cloudwatch_metric_alarm" "recover" {
  alarm_name          = "${var.project}-airflow-recover"
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed_System"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  dimensions = {
    InstanceId = aws_instance.airflow.id
  }
  alarm_actions = ["arn:aws:automate:${var.region}:ec2:recover"]
}
