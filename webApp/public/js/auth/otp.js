const inputs = document.querySelectorAll("input");

// loop the inputs field
inputs.forEach((input, idx) => {
    // condition when user is filling the field
    input.addEventListener('input', (e) => {
        if(e.target.value.length === e.target.maxLength && idx < inputs.length - 1){
            inputs[idx + 1].focus();
        };
    });

    // condition when user is entering backspace
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace' && e.target.value == "" && idx > 0){
            inputs[idx - 1].focus();
        }
    })
})