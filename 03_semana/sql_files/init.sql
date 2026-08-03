-- 1. Creación de Tablas
-- ------------------------------------------

CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY,
    CustomerName VARCHAR(100),
    City VARCHAR(50),
    Country VARCHAR(50)
);

CREATE TABLE Orders (
    OrderID INT PRIMARY KEY,
    CustomerID INT,
    OrderDate DATE,
    Amount DECIMAL(10, 2),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

CREATE TABLE Accounts (
    AccountID INT PRIMARY KEY,
    Balance DECIMAL(10, 2)
);

-- 2. Inserción de Datos de Prueba
-- ------------------------------------------

INSERT INTO Customers (CustomerID, CustomerName, City, Country) VALUES
(1, 'Alfreds Futterkiste', 'Berlin', 'Germany'),
(2, 'Ana Trujillo Emparedados', 'Mexico D.F.', 'Mexico'),
(3, 'Antonio Moreno Taqueria', 'Mexico D.F.', 'Mexico'),
(4, 'Around the Horn', 'London', 'UK'),
(5, 'Berglunds snabbkop', 'Lulea', 'Sweden'),
(6, 'Blauer See Delikatessen', 'Mannheim', 'Germany'),
(7, 'Centro comercial Moctezuma', 'Mexico D.F.', 'Mexico');

INSERT INTO Orders (OrderID, CustomerID, OrderDate, Amount) VALUES
(10308, 2, '2023-09-18', 150.00),
(10309, 3, '2023-09-19', 250.00),
(10310, 6, '2023-09-20', 300.00),
(10311, 1, '2023-09-20', 100.00),
(10312, 1, '2023-09-21', 400.00),
(10313, 4, '2023-09-22', 120.00);

INSERT INTO Accounts (AccountID, Balance) VALUES
(1, 1000.00),
(2, 500.00),
(3, 1500.00);