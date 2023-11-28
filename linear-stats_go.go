package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
)

func ReadFile(filename string) ([]float64, []float64) {
	var data_x []float64
	var data_y []float64
	var i int
	file, _ := os.Open(filename)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		tmp := scanner.Text()
		convert_tmp, _ := strconv.ParseFloat(tmp, 64)
		data_x = append(data_x, float64(i))
		data_y = append(data_y, convert_tmp)
		i++
	}
	return data_x, data_y
}

// Calc Average
func Average(data []float64) float64 {
	var average float64
	for i := range data {
		average += float64(data[i])
	}
	tmp := average / float64(len(data))
	return tmp
}

// Calc Variance
func Variance(data []float64, mean float64) float64 {
	var n float64
	for i := range data {
		n += math.Pow((data[i] - mean), 2)
	}
	return float64(1) / float64(len(data)) * n
}

// Calc covariance
func Covariance(data_x, data_y []float64, avg_x, avg_y float64) float64 {
	var k float64
	for i := range data_x {
		k += (data_x[i] - avg_x) * (data_y[i] - avg_y)
	}
	return (float64(1) / float64(len(data_x))) * k
}

// Calc linear stats
func Coeff(x, y []float64, m_x, m_y float64) (float64, float64) {
	n := float64(len(x))

	var k float64
	for i := range x {
		k += x[i] * y[i]
	}
	cross_dev := k - n*m_y*m_x

	var l float64
	for i := range x {
		l += x[i] * x[i]
	}
	dev_x := l - n*m_x*m_x

	a := cross_dev / dev_x
	b := m_y - a*m_x

	return a, b
}

// Calc Pearson Correlation
func Pcc(cov, v_x, v_y float64) float64 {
	return cov / (math.Sqrt(v_x * v_y))
}

func main() {
	args := os.Args[1:]
	if len(args) != 1 {
		return
	}

	data_x, data_y := ReadFile(args[0])
	m_x := Average(data_x)
	m_y := Average(data_y)

	a, b := Coeff(data_x, data_y, m_x, m_y)

	cov := Covariance(data_x, data_y, m_x, m_y)
	var_x := Variance(data_x, m_x)
	var_y := Variance(data_y, m_y)

	p := Pcc(cov, var_x, var_y)

	fmt.Printf("Linear Regression Line: y = %.6fx + %.6f\n", a, b)
	fmt.Printf("Pearson Correlation Coefficient: %.10f\n", p)
}
