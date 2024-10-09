const data = {
  labels: ['Churn', 'Tidak Churn'],
  datasets: [
    {
      label: 'Churn Dataset',
      data: [46, 54], // Churn 46, Tidak Churn 54
      backgroundColor: [ 'rgba(255, 99, 132, 0.5)','rgba(54, 162, 235, 0.5)'], // Merah untuk Churn, Biru untuk Tidak Churn
      borderColor: [ 'rgba(255, 99, 132, 1)','rgba(54, 162, 235, 1)'], 
    }
  ]
};

new Chart(document.getElementById('churnChart2'), {
  type: 'doughnut',
  data: data,
  options: {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#FFFFFF' // Label warna putih
        }
      }
    }
  }
});
