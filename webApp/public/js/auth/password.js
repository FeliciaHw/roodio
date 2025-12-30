const showPassBtn = document.getElementById("showPass");
const input = document.getElementById("password");
const closeEye = document.getElementById("eye-closed");
const openEye = document.getElementById("eye-open");
const classChange = 'invisible';

// change the behaviour of password field (show password)
showPassBtn.addEventListener("click", () => {
    if (!closeEye.classList.contains(classChange)){
        closeEye.classList.add(classChange);
        openEye.classList.remove(classChange);
        input.type = 'text';
    } else {
        closeEye.classList.remove(classChange);
        openEye.classList.add(classChange);
        input.type = 'password';
    }
});

// customize behaviour show password for register page
const btnConfirm = document.getElementById("showPassConfirm");
const altInput = document.getElementById("passwordConfirm");
const altCloseEye = document.getElementById("eye-closed-pass-confirm");
const altOpenEye = document.getElementById("eye-open-pass-confirm");

if (btnConfirm != null){
    btnConfirm.addEventListener("click", () => {
        if (!altCloseEye.classList.contains(classChange)){
            altCloseEye.classList.add(classChange);
            altOpenEye.classList.remove(classChange);
            altInput.type = 'text';
        } else {
            altCloseEye.classList.remove(classChange);
            altOpenEye.classList.add(classChange);
            altInput.type = 'password';
        }
    });
}