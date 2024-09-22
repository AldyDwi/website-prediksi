new Chart(document.getElementById('churnChart'), {
    type: 'bar',
    data: {
        labels: ['Red', 'Blue'],
        datasets: [{
            label: '# of Votes',
            data: [12, 19,],
            borderWidth: 1
        }]
    },
    options: {
        indexAxis: 'y',

    },
});