import subprocess
import os
import sys

def rodar_pipeline():
    print(f"\nINICIANDO PIPELINE DE MACHINE LEARNING (MLOps)\n")
    
    scripts_pipeline = [
        "data_collection.py",
        "pre_processing.py",
        "train_model.py",
        "validacao_local.py"
    ]

    python_exec = sys.executable

    for script in scripts_pipeline:
        print(f"\n Iniciando estágio: {script}...")
        
        resultado = subprocess.run([python_exec, script])
        
        if resultado.returncode != 0:
            print(f"\nERRO CRÍTICO: Falha ao executar '{script}'.")
            print(f"\nPipeline abortado para evitar dados corrompidos nas próximas etapas.\n")
            return
            
        print(f"Estágio '{script}' concluído com sucesso!")

    print(f"\nPIPELINE FINALIZADO COM SUCESSO!\n")


if __name__ == "__main__":
    rodar_pipeline()