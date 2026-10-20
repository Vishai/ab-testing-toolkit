variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for every resource. Lowercase, since it is part of the bucket name."
  type        = string
  default     = "abkit"

  validation {
    condition     = can(regex("^[a-z0-9-]{3,30}$", var.project))
    error_message = "Use 3 to 30 lowercase letters, digits or dashes."
  }
}

variable "vpc_id" {
  description = "VPC to run in."
  type        = string
}

variable "subnet_id" {
  description = "Public subnet for the instance (it pulls images and the repo from the internet)."
  type        = string
}

variable "allowed_cidrs" {
  description = "Who can reach the Airflow UI (8080) and dashboard (8501). Keep this narrow."
  type        = list(string)

  validation {
    condition     = !contains(var.allowed_cidrs, "0.0.0.0/0")
    error_message = "Don't open Airflow to the whole internet; list your office or VPN ranges."
  }
}

variable "instance_type" {
  description = "Four MCMC chains run in parallel, so four vCPUs is a good minimum."
  type        = string
  default     = "t3.xlarge"
}

variable "disk_gb" {
  description = "Root volume size. Docker images plus run artifacts."
  type        = number
  default     = 40
}

variable "repo_url" {
  description = "Git repository to deploy."
  type        = string
  default     = "https://github.com/Vishai/bayesian-experimentation-uplift-platform.git"
}

variable "repo_ref" {
  description = "Branch, tag or commit to deploy."
  type        = string
  default     = "main"
}

variable "force_destroy_bucket" {
  description = "Allow terraform destroy to delete a bucket that still has readouts in it."
  type        = bool
  default     = false
}
