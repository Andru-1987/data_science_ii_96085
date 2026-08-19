-- =====================================================
-- DATA WRANGLING II E-COMMERCE
-- Tabla de transacciones de tienda online
-- =====================================================

-- Eliminar tabla si existe
DROP TABLE IF EXISTS ecommerce_transacciones;

-- Crear tabla principal
CREATE TABLE ecommerce_transacciones (
    id_transaccion INT NOT NULL,
    cliente_zona VARCHAR(20) NOT NULL,
    categoria VARCHAR(20) NOT NULL,
    monto_usd DECIMAL(10,2) NOT NULL,
    estado_envio VARCHAR(20) NOT NULL,
    fecha_transaccion DATE DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id_transaccion, fecha_transaccion)
);

-- =====================================================
-- INSERCIÓN DE 100 REGISTROS
-- =====================================================

INSERT INTO ecommerce_transacciones (id_transaccion, cliente_zona, categoria, monto_usd, estado_envio, fecha_transaccion) VALUES
(1001, 'Norte', 'Electrónica', 250.00, 'Entregado', '2026-01-15'),
(1002, 'Sur', 'Hogar', 50.00, 'En tránsito', '2026-01-15'),
(1003, 'Este', 'Ropa', 30.00, 'Entregado', '2026-01-16'),
(1004, 'Norte', 'Electrónica', 120.00, 'Devuelto', '2026-01-16'),
(1005, 'Oeste', 'Ropa', 45.00, 'Entregado', '2026-01-17'),
(1006, 'Norte', 'Hogar', 180.00, 'Entregado', '2026-01-17'),
(1007, 'Sur', 'Electrónica', 320.00, 'Entregado', '2026-01-18'),
(1008, 'Este', 'Hogar', 75.00, 'En tránsito', '2026-01-18'),
(1009, 'Oeste', 'Electrónica', 210.00, 'Entregado', '2026-01-19'),
(1010, 'Norte', 'Ropa', 65.00, 'Entregado', '2026-01-19'),
(1011, 'Sur', 'Ropa', 89.00, 'Devuelto', '2026-01-20'),
(1012, 'Este', 'Electrónica', 450.00, 'Entregado', '2026-01-20'),
(1013, 'Oeste', 'Hogar', 95.00, 'Entregado', '2026-01-21'),
(1014, 'Norte', 'Electrónica', 150.00, 'En tránsito', '2026-01-21'),
(1015, 'Sur', 'Hogar', 220.00, 'Entregado', '2026-01-22'),
(1016, 'Este', 'Ropa', 40.00, 'Entregado', '2026-01-22'),
(1017, 'Oeste', 'Electrónica', 180.00, 'Devuelto', '2026-01-23'),
(1018, 'Norte', 'Hogar', 130.00, 'Entregado', '2026-01-23'),
(1019, 'Sur', 'Electrónica', 500.00, 'Entregado', '2026-01-24'),
(1020, 'Este', 'Hogar', 60.00, 'En tránsito', '2026-01-24'),
(1021, 'Oeste', 'Ropa', 55.00, 'Entregado', '2026-01-25'),
(1022, 'Norte', 'Electrónica', 280.00, 'Entregado', '2026-01-25'),
(1023, 'Sur', 'Ropa', 70.00, 'Entregado', '2026-01-26'),
(1024, 'Este', 'Electrónica', 390.00, 'Devuelto', '2026-01-26'),
(1025, 'Oeste', 'Hogar', 110.00, 'Entregado', '2026-01-27'),
(1026, 'Norte', 'Ropa', 85.00, 'En tránsito', '2026-01-27'),
(1027, 'Sur', 'Electrónica', 260.00, 'Entregado', '2026-01-28'),
(1028, 'Este', 'Hogar', 145.00, 'Entregado', '2026-01-28'),
(1029, 'Oeste', 'Electrónica', 310.00, 'Entregado', '2026-01-29'),
(1030, 'Norte', 'Hogar', 200.00, 'Devuelto', '2026-01-29'),
(1031, 'Sur', 'Ropa', 35.00, 'Entregado', '2026-01-30'),
(1032, 'Este', 'Electrónica', 420.00, 'Entregado', '2026-01-30'),
(1033, 'Oeste', 'Hogar', 90.00, 'En tránsito', '2026-02-01'),
(1034, 'Norte', 'Electrónica', 170.00, 'Entregado', '2026-02-01'),
(1035, 'Sur', 'Hogar', 240.00, 'Entregado', '2026-02-02'),
(1036, 'Este', 'Ropa', 48.00, 'Devuelto', '2026-02-02'),
(1037, 'Oeste', 'Electrónica', 350.00, 'Entregado', '2026-02-03'),
(1038, 'Norte', 'Ropa', 92.00, 'Entregado', '2026-02-03'),
(1039, 'Sur', 'Electrónica', 480.00, 'En tránsito', '2026-02-04'),
(1040, 'Este', 'Hogar', 160.00, 'Entregado', '2026-02-04'),
(1041, 'Oeste', 'Ropa', 38.00, 'Entregado', '2026-02-05'),
(1042, 'Norte', 'Electrónica', 300.00, 'Devuelto', '2026-02-05'),
(1043, 'Sur', 'Hogar', 195.00, 'Entregado', '2026-02-06'),
(1044, 'Este', 'Ropa', 52.00, 'Entregado', '2026-02-06'),
(1045, 'Oeste', 'Electrónica', 225.00, 'Entregado', '2026-02-07'),
(1046, 'Norte', 'Hogar', 175.00, 'En tránsito', '2026-02-07'),
(1047, 'Sur', 'Ropa', 78.00, 'Entregado', '2026-02-08'),
(1048, 'Este', 'Electrónica', 560.00, 'Entregado', '2026-02-08'),
(1049, 'Oeste', 'Hogar', 105.00, 'Devuelto', '2026-02-09'),
(1050, 'Norte', 'Electrónica', 140.00, 'Entregado', '2026-02-09'),
(1051, 'Sur', 'Hogar', 215.00, 'Entregado', '2026-02-10'),
(1052, 'Este', 'Ropa', 62.00, 'En tránsito', '2026-02-10'),
(1053, 'Oeste', 'Electrónica', 270.00, 'Entregado', '2026-02-11'),
(1054, 'Norte', 'Ropa', 55.00, 'Entregado', '2026-02-11'),
(1055, 'Sur', 'Electrónica', 410.00, 'Devuelto', '2026-02-12'),
(1056, 'Este', 'Hogar', 135.00, 'Entregado', '2026-02-12'),
(1057, 'Oeste', 'Ropa', 72.00, 'Entregado', '2026-02-13'),
(1058, 'Norte', 'Electrónica', 290.00, 'Entregado', '2026-02-13'),
(1059, 'Sur', 'Hogar', 165.00, 'En tránsito', '2026-02-14'),
(1060, 'Este', 'Electrónica', 380.00, 'Entregado', '2026-02-14'),
(1061, 'Oeste', 'Hogar', 88.00, 'Entregado', '2026-02-15'),
(1062, 'Norte', 'Ropa', 43.00, 'Devuelto', '2026-02-15'),
(1063, 'Sur', 'Electrónica', 330.00, 'Entregado', '2026-02-16'),
(1064, 'Este', 'Hogar', 155.00, 'Entregado', '2026-02-16'),
(1065, 'Oeste', 'Ropa', 68.00, 'En tránsito', '2026-02-17'),
(1066, 'Norte', 'Electrónica', 190.00, 'Entregado', '2026-02-17'),
(1067, 'Sur', 'Hogar', 230.00, 'Entregado', '2026-02-18'),
(1068, 'Este', 'Ropa', 47.00, 'Devuelto', '2026-02-18'),
(1069, 'Oeste', 'Electrónica', 295.00, 'Entregado', '2026-02-19'),
(1070, 'Norte', 'Hogar', 210.00, 'Entregado', '2026-02-19'),
(1071, 'Sur', 'Ropa', 82.00, 'Entregado', '2026-02-20'),
(1072, 'Este', 'Electrónica', 490.00, 'En tránsito', '2026-02-20'),
(1073, 'Oeste', 'Hogar', 115.00, 'Entregado', '2026-02-21'),
(1074, 'Norte', 'Electrónica', 160.00, 'Devuelto', '2026-02-21'),
(1075, 'Sur', 'Hogar', 250.00, 'Entregado', '2026-02-22'),
(1076, 'Este', 'Ropa', 33.00, 'Entregado', '2026-02-22'),
(1077, 'Oeste', 'Electrónica', 340.00, 'Entregado', '2026-02-23'),
(1078, 'Norte', 'Ropa', 97.00, 'En tránsito', '2026-02-23'),
(1079, 'Sur', 'Electrónica', 440.00, 'Entregado', '2026-02-24'),
(1080, 'Este', 'Hogar', 125.00, 'Entregado', '2026-02-24'),
(1081, 'Oeste', 'Ropa', 41.00, 'Devuelto', '2026-02-25'),
(1082, 'Norte', 'Electrónica', 310.00, 'Entregado', '2026-02-25'),
(1083, 'Sur', 'Hogar', 185.00, 'Entregado', '2026-02-26'),
(1084, 'Este', 'Electrónica', 370.00, 'En tránsito', '2026-02-26'),
(1085, 'Oeste', 'Hogar', 98.00, 'Entregado', '2026-02-27'),
(1086, 'Norte', 'Ropa', 58.00, 'Entregado', '2026-02-27'),
(1087, 'Sur', 'Electrónica', 510.00, 'Devuelto', '2026-02-28'),
(1088, 'Este', 'Hogar', 140.00, 'Entregado', '2026-02-28'),
(1089, 'Oeste', 'Ropa', 75.00, 'Entregado', '2026-03-01'),
(1090, 'Norte', 'Electrónica', 230.00, 'Entregado', '2026-03-01'),
(1091, 'Sur', 'Hogar', 205.00, 'En tránsito', '2026-03-02'),
(1092, 'Este', 'Ropa', 44.00, 'Entregado', '2026-03-02'),
(1093, 'Oeste', 'Electrónica', 285.00, 'Entregado', '2026-03-03'),
(1094, 'Norte', 'Hogar', 195.00, 'Devuelto', '2026-03-03'),
(1095, 'Sur', 'Ropa', 88.00, 'Entregado', '2026-03-04'),
(1096, 'Este', 'Electrónica', 460.00, 'Entregado', '2026-03-04'),
(1097, 'Oeste', 'Hogar', 108.00, 'Entregado', '2026-03-05'),
(1098, 'Norte', 'Electrónica', 270.00, 'En tránsito', '2026-03-05'),
(1099, 'Sur', 'Hogar', 238.00, 'Entregado', '2026-03-06'),
(1100, 'Este', 'Ropa', 36.00, 'Entregado', '2026-03-06');

-- =====================================================
-- CONSULTAS DE VERIFICACIÓN
-- =====================================================

-- Ver total de registros insertados
SELECT COUNT(*) AS total_registros FROM ecommerce_transacciones;

-- Ver distribución por zona
SELECT cliente_zona, COUNT(*) AS cantidad 
FROM ecommerce_transacciones 
GROUP BY cliente_zona 
ORDER BY cantidad DESC;

-- Ver distribución por categoría
SELECT categoria, COUNT(*) AS cantidad 
FROM ecommerce_transacciones 
GROUP BY categoria 
ORDER BY cantidad DESC;

-- Ver distribución por estado de envío
SELECT estado_envio, COUNT(*) AS cantidad 
FROM ecommerce_transacciones 
GROUP BY estado_envio 
ORDER BY cantidad DESC;

-- Ticket promedio por zona y categoría (similar al pivot de pandas)
SELECT 
    cliente_zona,
    ROUND(AVG(CASE WHEN categoria = 'Electrónica' THEN monto_usd ELSE NULL END), 2) AS promedio_electronica,
    ROUND(AVG(CASE WHEN categoria = 'Hogar' THEN monto_usd ELSE NULL END), 2) AS promedio_hogar,
    ROUND(AVG(CASE WHEN categoria = 'Ropa' THEN monto_usd ELSE NULL END), 2) AS promedio_ropa
FROM ecommerce_transacciones
GROUP BY cliente_zona
ORDER BY cliente_zona;

-- Top 10 transacciones con mayor monto
SELECT id_transaccion, cliente_zona, categoria, monto_usd, estado_envio
FROM ecommerce_transacciones
ORDER BY monto_usd DESC
LIMIT 10;

-- Ventas VIP (>100 USD y entregadas)
SELECT id_transaccion, cliente_zona, categoria, monto_usd
FROM ecommerce_transacciones
WHERE estado_envio = 'Entregado' AND monto_usd > 100
ORDER BY monto_usd DESC;