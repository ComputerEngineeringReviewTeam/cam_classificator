async function sendDeleteRequest(datapoint_id) {
    fetch('/cam/training_data/' + datapoint_id, {
            method: 'DELETE',
        })
        .then(r => {
            if (r.status === 204) {
                console.log('Deleted datapoint ...redirecting')
                window.location.href = '/cam/training_data/all'
            } else {
                console.log('Error:', r)
            }
        })
        .catch(error => console.log('Error:', error))
}