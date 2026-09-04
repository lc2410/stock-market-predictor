terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 5.0"
    }
  }
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# Fetch Availability Domain
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# Fetch latest Ubuntu 22.04 ARM image
data "oci_core_images" "ubuntu_arm" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

# Network Configuration
resource "oci_core_vcn" "app_vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "marketlens-vcn"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.app_vcn.id
  enabled        = true
}

resource "oci_core_default_route_table" "public_route" {
  manage_default_resource_id = oci_core_vcn.app_vcn.default_route_table_id
  route_rules {
    network_entity_id = oci_core_internet_gateway.igw.id
    destination       = "0.0.0.0/0"
  }
}

resource "oci_core_default_security_list" "public_security" {
  manage_default_resource_id = oci_core_vcn.app_vcn.default_security_list_id

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  ingress_security_rules {
    protocol = "6" # TCP
    source   = "0.0.0.0/0"
    tcp_options {
      min = 22
      max = 22
    }
  }

  ingress_security_rules {
    protocol = "6" # TCP
    source   = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }
}

resource "oci_core_subnet" "public_subnet" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.app_vcn.id
  cidr_block     = "10.0.1.0/24"
}

# Compute Instance
resource "oci_core_instance" "app_server" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  display_name        = "marketlens-server"
  shape               = "VM.Standard.A1.Flex"

  shape_config {
    ocpus         = 2
    memory_in_gbs = 12
  }

  source_details {
    source_id   = data.oci_core_images.ubuntu_arm.images[0].id
    source_type = "image"
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public_subnet.id
    assign_public_ip = true
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data           = base64encode(<<-EOF
      #!/bin/bash
      # Open OS Firewall
      iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
      netfilter-persistent save

      # Install Dependencies
      apt-get update
      apt-get install -y python3-pip python3-venv git

      # Clone and Setup App
      cd /home/ubuntu
      git clone https://github.com/lc2410/stock-market-predictor.git
      cd stock-market-predictor
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt

      # Create Systemd Service for Gunicorn
      cat << 'SERVICE' > /etc/systemd/system/marketlens.service
      [Unit]
      Description=Gunicorn instance to serve MarketLens
      After=network.target

      [Service]
      User=root
      Group=www-data
      WorkingDirectory=/home/ubuntu/stock-market-predictor/backend
      Environment="PATH=/home/ubuntu/stock-market-predictor/venv/bin"
      ExecStart=/home/ubuntu/stock-market-predictor/venv/bin/gunicorn -w 4 --worker-class gthread --threads 10 -b 127.0.0.1:5001 --timeout 120 app:app

      [Install]
      WantedBy=multi-user.target
      SERVICE

      systemctl start marketlens
      systemctl enable marketlens


    EOF
    )
  }

}

output "public_ip" {
  value = oci_core_instance.app_server.public_ip
}
resource "oci_database_autonomous_database" "db" {
  admin_password = var.db_password
  compartment_id = var.compartment_ocid
  db_name = "marketlensdb"

  db_workload = "OLTP"
  display_name = "MarketLens_DB"

  is_free_tier = true

  is_mtls_connection_required = true

  lifecycle {
    ignore_changes = [cpu_core_count, data_storage_size_in_tbs]
  }



}

resource "oci_database_autonomous_database_wallet" "wallet" {
  autonomous_database_id = oci_database_autonomous_database.db.id
  password               = var.db_password
  base64_encode_content  = true
}

output "db_connection_strings" {
  value = oci_database_autonomous_database.db.connection_strings
}

output "exact_db_dsn" {
  description = "Copy this EXACT string to your GitHub DB_DSN Secret"
  value       = "${oci_database_autonomous_database.db.db_name}_high"
}

output "db_wallet_base64" {
  description = "Base64 encoded wallet. Run `terraform output -raw db_wallet_base64 | pbcopy` and paste into GitHub Secret DB_WALLET_BASE64"
  value       = oci_database_autonomous_database_wallet.wallet.content
  sensitive   = true
}
