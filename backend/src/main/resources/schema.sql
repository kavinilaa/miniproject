CREATE DATABASE IF NOT EXISTS miniproject;

USE miniproject;

CREATE TABLE IF NOT EXISTS doctors (
    doctor_id   VARCHAR(20)  PRIMARY KEY,
    full_name   VARCHAR(100) NOT NULL,
    email       VARCHAR(150) NOT NULL UNIQUE,
    password    VARCHAR(255) NOT NULL
);
