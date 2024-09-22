const labels = ['Churn']; // Label diubah menjadi array dengan satu nilai 'Churn'

// Data churn dengan dua kategori: Tidak (biru) dan Iya (merah)
const data = {
  labels: labels,
  datasets: [
    {
      label: 'Tidak',
      data: [80], // Persentase "Tidak"
      borderColor: 'rgba(54, 162, 235, 1)',
      backgroundColor: 'rgba(54, 162, 235, 0.5)',
      borderWidth: 2,
      borderRadius: 5,
      borderSkipped: false,
    },
    {
      label: 'Iya',
      data: [20], // Persentase "Iya"
      borderColor: 'rgba(255, 99, 132, 1)',
      backgroundColor: 'rgba(255, 99, 132, 0.5)',
      borderWidth: 2,
      borderRadius: 5,
      borderSkipped: false,
    },
  ],
};

new Chart(document.getElementById('potensiChurn'), {
  type: 'bar',
  data: data,
  options: {
    indexAxis: 'y', // Bar horizontal
    elements: {
      bar: {
        borderWidth: 2,
      },
    },
    responsive: true,
    plugins: {
      legend: {
        labels: {
          color: 'white', // Mengatur warna tulisan legend menjadi putih
          font: {
            size: 14, // Ukuran font untuk legend
          },
        },
      },
      datalabels: {
        display: true, // Menampilkan nilai persentase di setiap batang
        color: 'white', // Mengubah warna data label menjadi putih
        align: 'center',
        anchor: 'center',
        formatter: (value) => value + '%', // Menampilkan persentase
      },
    },
    scales: {
      x: {
        stacked: true,
        ticks: {
          display: false, // Menghilangkan label angka di sumbu X
        },
        grid: {
          display: false, // Menghilangkan garis grid di sumbu X
        },
      },
      y: {
        stacked: true,
        ticks: {
          display: false, // Menghilangkan label angka di sumbu Y
        },
        grid: {
          display: false, // Menghilangkan garis grid di sumbu Y
        },
      },
    },
  },
  plugins: [ChartDataLabels], // Plugin untuk menampilkan data di dalam bar
});
