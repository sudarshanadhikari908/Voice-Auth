const buttons = document.querySelectorAll(".tab-btn");
const change = document.querySelector("#container");
const forms = document.querySelectorAll(".content");

change.addEventListener('click',(e)=>{
    const id = e.target.dataset.id;
    if(id){
        buttons.forEach((btn)=>{
            btn.classList.remove('selected');
            e.target.classList.add('selected');
        });
       forms.forEach((form)=>{
           form.classList.remove('active');
           
       })
       const element = document.getElementById(id);
       element.classList.add('active');
    }

})

