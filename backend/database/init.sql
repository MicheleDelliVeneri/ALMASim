-- Initial database setup for ALMASim
-- This file is executed by PostgreSQL on first container startup

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for better query performance (tables will be created by SQLAlchemy)
-- These will be applied after the tables are created by the application

-- Note: Table creation is handled by SQLAlchemy models
-- This file is for any additional setup needed at database initialization time
