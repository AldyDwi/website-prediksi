// new Chart(document.getElementById('churnChart'), {
//     type: 'bar',
//     data: {
//         labels: ['Red', 'Blue'],
//         datasets: [{
//             label: '# of Votes',
//             data: [12, 19,],
//             borderWidth: 1
//         }]
//     },
//     options: {
//         indexAxis: 'y',

//     },
// });
new Chart(document.getElementById('churnChart'), {
  type: 'bar',
  data: {
    labels: ['Churn', 'Tidak Churn'],
    datasets: [
      {
        data: [46, 54],
        backgroundColor: ['rgba(54, 162, 235, 0.5)', 'rgba(255, 99, 132, 0.5)'], // Mengatur warna bar
        borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'], // Mengatur warna border bar
        borderWidth: 1,
      },
    ],
  },
  options: {
    indexAxis: 'y',
    scales: {
      y: {
        ticks: {
          color: 'white', // Warna label pada sumbu y
        },
      },
      x: {
        ticks: {
          color: 'white', // Warna label pada sumbu x
        },
      },
    },
    plugins: {
      legend: {
        display: false, // Menyembunyikan legenda karena hanya dua kategori
      },
    },
  },
});
