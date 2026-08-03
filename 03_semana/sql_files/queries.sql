-- JOINS

-- **Objetivo:** Combinar tablas y aplicar múltiples condiciones lógicas.
-- **Consigna:** Escribe una consulta SQL que devuelva el nombre del cliente, la ciudad y el monto de la orden, pero solo para aquellos clientes de 'Mexico' cuyas órdenes superen los 200.00.
-- **Solución esperada:**


SELECT c.CustomerName, c.City, o.Amount
FROM Customers c
INNER JOIN Orders o ON c.CustomerID = o.CustomerID
WHERE c.Country = 'Mexico' AND o.Amount > 200.00;

--  Agrupaciones y HAVING

-- **Objetivo:** Utilizar funciones de agregación y filtrar grupos.
-- **Consigna:** Calcula el monto total de ventas (`Amount`) por cada país (`Country`). Muestra únicamente aquellos países donde el monto total acumulado de sus órdenes sea estrictamente mayor a 300.00.
-- **Solución esperada:**

SELECT c.Country, SUM(o.Amount) AS TotalSales
FROM Customers c
INNER JOIN Orders o ON c.CustomerID = o.CustomerID
GROUP BY c.Country
HAVING SUM(o.Amount) > 300.00;

-- Transacciones TCL

-- **Objetivo:** Garantizar la integridad de los datos en operaciones múltiples.
-- **Consigna:** Utilizando la tabla `Accounts`, simula una transferencia bancaria de 200.00 desde la cuenta con `AccountID = 3` hacia la cuenta con `AccountID = 2`. Asegúrate de utilizar las sentencias de transacción adecuadas para que ambas operaciones se apliquen de forma segura.
-- **Solución esperada:**

BEGIN;
UPDATE Accounts SET Balance = Balance - 200.00 WHERE AccountID = 3;
UPDATE Accounts SET Balance = Balance + 200.00 WHERE AccountID = 2;
COMMIT;
